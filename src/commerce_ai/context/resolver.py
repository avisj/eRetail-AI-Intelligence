"""Deterministic Context Resolution Engine (Phase 7E).

Implements conversational cue resolution, entity inheritance, filter override,
time range shifting, user corrections, ambiguity resolution, domain compatibility pruning,
and security sanitization.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from commerce_ai.context.enums import (
    CompatibilityStatus,
    ContextCategory,
    ContextConfidence,
    ContextProvenance,
    ContextScope,
)
from commerce_ai.context.governance import sanitize_input_for_injection
from commerce_ai.context.schemas import (
    AmbiguityStateValue,
    ContextItem,
    ContextResolutionResult,
    ConversationContextSnapshot,
    CorrectionEvent,
    EntityContextValue,
    FilterContextValue,
    TimeRangeContextValue,
)


def calculate_calendar_period(preset: str, anchor_str: str) -> Tuple[str, str, Dict[str, Any]]:
    """Calculate true calendar period boundaries for relative temporal expressions.

    Supported presets:
    - PRIOR_MONTH: First day through last day of previous calendar month
    - PRIOR_QUARTER: First day through last day of previous calendar quarter
    - PRIOR_YEAR: Jan 1 through Dec 31 of previous calendar year
    - PRIOR_WEEK: Monday through Sunday of the week preceding anchor date's calendar week
    - YESTERDAY: Exact calendar date immediately preceding anchor date
    """
    try:
        anchor_dt = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    except Exception:
        anchor_dt = date(2026, 6, 30)

    year = anchor_dt.year
    month = anchor_dt.month

    if preset == "PRIOR_MONTH":
        if month == 1:
            prev_year = year - 1
            prev_month = 12
        else:
            prev_year = year
            prev_month = month - 1
        last_day = calendar.monthrange(prev_year, prev_month)[1]
        start_date = date(prev_year, prev_month, 1)
        end_date = date(prev_year, prev_month, last_day)

    elif preset == "PRIOR_QUARTER":
        current_q = (month - 1) // 3 + 1
        if current_q == 1:
            prev_q = 4
            prev_year = year - 1
        else:
            prev_q = current_q - 1
            prev_year = year
        start_month = (prev_q - 1) * 3 + 1
        end_month = prev_q * 3
        last_day = calendar.monthrange(prev_year, end_month)[1]
        start_date = date(prev_year, start_month, 1)
        end_date = date(prev_year, end_month, last_day)

    elif preset == "PRIOR_YEAR":
        prev_year = year - 1
        start_date = date(prev_year, 1, 1)
        end_date = date(prev_year, 12, 31)

    elif preset == "PRIOR_WEEK":
        # Calendar week: Monday through Sunday immediately preceding current week
        curr_monday = anchor_dt - timedelta(days=anchor_dt.weekday())
        start_date = curr_monday - timedelta(days=7)
        end_date = curr_monday - timedelta(days=1)

    elif preset == "YESTERDAY":
        yest = anchor_dt - timedelta(days=1)
        start_date = yest
        end_date = yest

    else:
        end_date = anchor_dt - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    details = {
        "preset": preset,
        "anchor_as_of_date": anchor_str,
        "shifted_start_date": start_str,
        "shifted_end_date": end_str,
        "calendar_semantics": True,
    }
    return start_str, end_str, details


# Entity extraction regex patterns
_SKU_REGEX = re.compile(r"\b(sku[_-]\w+|sku\d+)\b", re.IGNORECASE)
_WH_REGEX = re.compile(r"\b(wh[_-]\w+|wh\d+|warehouse\s+([A-Za-z0-9_-]+))\b", re.IGNORECASE)
_CHANNEL_REGEX = re.compile(r"\b(ONLINE|RETAIL|MARKETPLACE|STORE|WHOLESALE)\b", re.IGNORECASE)
_SUPPLIER_REGEX = re.compile(r"\b(SUP[_-]?[A-Za-z0-9]+|SUPPLIER[_-]?[A-Za-z0-9]+)\b", re.IGNORECASE)

# Follow-up and pronoun cues
_PRONOUN_CUES = re.compile(r"\b(it|its|that|this|same|the)\s+(sku|product|item|warehouse|channel|location)\b|\b(its|it)\b", re.IGNORECASE)

# User correction patterns
_CORRECTION_PATTERNS = [
    re.compile(r"\b(?:actually|correction|wait|no),?\s+(?:i\s+meant|not)\s+(?:warehouse\s+)?(WH[_-]?[A-Za-z0-9]+)\b", re.IGNORECASE),
    re.compile(r"\b(?:actually|correction|wait|no),?\s+(?:i\s+meant|not)\s+(?:sku\s+)?(SKU[_-]?[A-Za-z0-9]+)\b", re.IGNORECASE),
    re.compile(r"\b(?:actually|correction|wait|no),?\s+(?:i\s+meant|not)\s+(?:channel\s+)?(ONLINE|RETAIL|MARKETPLACE|STORE|WHOLESALE)\b", re.IGNORECASE),
    re.compile(r"\b(?:forget|remove|clear|drop)\s+(?:the\s+)?(warehouse|channel|sku|supplier|filter)\b", re.IGNORECASE),
]

# Time shift cues
_TIME_SHIFT_PATTERNS = [
    (re.compile(r"\b(last\s+month|prior\s+month|previous\s+month)\b", re.IGNORECASE), "PRIOR_MONTH"),
    (re.compile(r"\b(last\s+quarter|prior\s+quarter|previous\s+quarter)\b", re.IGNORECASE), "PRIOR_QUARTER"),
    (re.compile(r"\b(last\s+year|prior\s+year|previous\s+year)\b", re.IGNORECASE), "PRIOR_YEAR"),
    (re.compile(r"\b(last\s+week|prior\s+week|previous\s+week)\b", re.IGNORECASE), "PRIOR_WEEK"),
    (re.compile(r"\b(yesterday)\b", re.IGNORECASE), "YESTERDAY"),
]

# Domain detection cues
_DOMAIN_CUES = [
    (re.compile(r"\b(inventory|stock|stockout|slow\s+moving|holding\s+cost|turnover)\b", re.IGNORECASE), "INVENTORY"),
    (re.compile(r"\b(margin|profit|ebitda|gross\s+margin|cogs|unit\s+economics)\b", re.IGNORECASE), "FINANCIAL"),
    (re.compile(r"\b(sales|revenue|units\s+sold|orders|order\s+count|volume)\b", re.IGNORECASE), "SALES"),
    (re.compile(r"\b(return|refund|reversal|return\s+rate|defect)\b", re.IGNORECASE), "RETURNS"),
    (re.compile(r"\b(supplier|procurement|lead\s+time|vendor|reorder|purchase\s+order)\b", re.IGNORECASE), "SUPPLIER"),
    (re.compile(r"\b(forecast|future\s+demand|predicted\s+sales|demand\s+projection)\b", re.IGNORECASE), "FORECASTING"),
    (re.compile(r"\b(pricing|elasticity|discount|markdown|promotion)\b", re.IGNORECASE), "PRICING"),
]


class ContextResolver:
    """Deterministic resolution engine for conversational follow-ups and context merging."""

    def resolve(
        self,
        question: str,
        snapshot: ConversationContextSnapshot,
        explicit_filters: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> ContextResolutionResult:
        """Resolve conversational follow-ups, entity/filter inheritance, corrections, and time shifts."""
        inherited_entities: Dict[str, Any] = {}
        overridden_entities: Dict[str, Any] = {}
        inherited_filters: Dict[str, Any] = {}
        overridden_filters: Dict[str, Any] = {}
        incompatible_pruned: Dict[str, Any] = {}
        corrections_applied: List[CorrectionEvent] = []
        lineage_events: List[Dict[str, Any]] = []
        ambiguity_resolved = False

        # 1. Security & Prompt Injection Sanitization
        sanitized_question, sec_violations = sanitize_input_for_injection(question)

        # 2. Check Ambiguity Resolution
        resolved_question = sanitized_question
        if snapshot.active_ambiguity and snapshot.active_ambiguity.status == "PENDING":
            amb = snapshot.active_ambiguity
            # Check if current question answers the missing fields
            extracted_val = None
            if "warehouse_id" in amb.missing_fields or "warehouse" in amb.missing_fields:
                wh_match = _WH_REGEX.search(sanitized_question)
                if wh_match:
                    extracted_val = (wh_match.group(2) if wh_match.group(2) else wh_match.group(1)).upper()
                    overridden_entities["warehouse_id"] = extracted_val
                    resolved_question = f"{amb.original_question} in warehouse {extracted_val}"
                    ambiguity_resolved = True
            elif "sku_id" in amb.missing_fields or "sku" in amb.missing_fields:
                sku_match = _SKU_REGEX.search(sanitized_question)
                if sku_match:
                    extracted_val = sku_match.group(1).upper()
                    overridden_entities["sku_id"] = extracted_val
                    resolved_question = f"{amb.original_question} for SKU {extracted_val}"
                    ambiguity_resolved = True

        # 3. Handle Explicit User Corrections
        correction_event = self._detect_correction(sanitized_question, snapshot)
        if correction_event:
            corrections_applied.append(correction_event)
            if correction_event.target_category == ContextCategory.ENTITY_CONTEXT:
                if correction_event.new_value is None:
                    # Filter removal
                    overridden_entities[correction_event.target_key] = None
                else:
                    overridden_entities[correction_event.target_key] = correction_event.new_value
            lineage_events.append({
                "operation": "USER_CORRECTION",
                "target_key": correction_event.target_key,
                "old_value": correction_event.previous_value,
                "new_value": correction_event.new_value,
                "reason": correction_event.reason,
            })

        # 4. Extract Explicit Entities from Question
        explicit_sku = None
        explicit_wh = None
        explicit_channel = None
        explicit_sup = None

        sku_m = _SKU_REGEX.search(sanitized_question)
        if sku_m:
            explicit_sku = sku_m.group(1).upper()
        wh_m = _WH_REGEX.search(sanitized_question)
        if wh_m:
            cand = (wh_m.group(2) if wh_m.group(2) else wh_m.group(1)).upper()
            _wh_exclusions = {"FILTER", "FILTERS", "SUMMARY", "RISK", "STOCK", "SALES", "INVENTORY", "PERFORMANCE", "LEVEL", "LEVELS", "DATA", "STATUS", "REBALANCING"}
            if cand not in _wh_exclusions and not (correction_event and correction_event.target_key == "warehouse_id" and correction_event.new_value is None):
                explicit_wh = cand
        ch_m = _CHANNEL_REGEX.search(sanitized_question)
        if ch_m:
            explicit_channel = ch_m.group(1).upper()
        sup_m = _SUPPLIER_REGEX.search(sanitized_question)
        if sup_m:
            explicit_sup = sup_m.group(1).upper()

        # 5. Infer Current Domain
        current_domain = None
        for pat, dom in _DOMAIN_CUES:
            if pat.search(sanitized_question):
                current_domain = dom
                break

        # 6. Entity Inheritance & Override Rules
        # SKU
        if explicit_sku:
            if snapshot.entities.sku_id and snapshot.entities.sku_id != explicit_sku:
                overridden_entities["sku_id"] = explicit_sku
            else:
                overridden_entities["sku_id"] = explicit_sku
        else:
            if snapshot.entities.sku_id and (
                _PRONOUN_CUES.search(sanitized_question)
                or current_domain is not None
                or not ambiguity_resolved
            ):
                inherited_entities["sku_id"] = snapshot.entities.sku_id

        # Warehouse
        if explicit_wh:
            if snapshot.entities.warehouse_id and snapshot.entities.warehouse_id != explicit_wh:
                overridden_entities["warehouse_id"] = explicit_wh
            else:
                overridden_entities["warehouse_id"] = explicit_wh
        else:
            if snapshot.entities.warehouse_id and (
                _PRONOUN_CUES.search(sanitized_question)
                or current_domain in ("INVENTORY", "SALES", None)
            ):
                inherited_entities["warehouse_id"] = snapshot.entities.warehouse_id

        # Channel
        if explicit_channel:
            if snapshot.entities.channel_id and snapshot.entities.channel_id != explicit_channel:
                overridden_entities["channel_id"] = explicit_channel
            else:
                overridden_entities["channel_id"] = explicit_channel
        else:
            if snapshot.entities.channel_id and (
                _PRONOUN_CUES.search(sanitized_question)
                or current_domain in ("SALES", "RETURNS", "FINANCIAL", None)
            ):
                inherited_entities["channel_id"] = snapshot.entities.channel_id

        # Supplier
        if explicit_sup:
            overridden_entities["supplier_id"] = explicit_sup
        else:
            if snapshot.entities.supplier_id and current_domain in ("SUPPLIER", "INVENTORY", None):
                inherited_entities["supplier_id"] = snapshot.entities.supplier_id

        # 7. Time Range Shifting
        time_shift_applied = None
        for pat, preset in _TIME_SHIFT_PATTERNS:
            if pat.search(sanitized_question):
                # Calculate shifted dates using exact calendar period semantics
                anchor = as_of_date or (
                    snapshot.filters.time_range.as_of_date if snapshot.filters.time_range else None
                ) or "2026-06-30"
                start_str, end_str, details = calculate_calendar_period(preset, anchor)
                time_shift_applied = details
                inherited_filters["date_from"] = start_str
                inherited_filters["date_to"] = end_str
                break

        # 8. Domain Switch & Semantic Compatibility Pruning
        if current_domain == "SUPPLIER":
            # Channel is completely incompatible with supplier queries
            if "channel_id" in inherited_entities:
                incompatible_pruned["channel_id"] = inherited_entities.pop("channel_id")
            if snapshot.entities.channel_id:
                incompatible_pruned["context_channel_id"] = snapshot.entities.channel_id
        elif current_domain == "INVENTORY":
            # Channel is incompatible with pure inventory queries
            if "channel_id" in inherited_entities:
                incompatible_pruned["channel_id"] = inherited_entities.pop("channel_id")

        # 9. Synthesize Resolved Question if Pronoun / Ellipsis Follow-up
        if not ambiguity_resolved:
            supplements: List[str] = []
            final_sku = overridden_entities.get("sku_id") or inherited_entities.get("sku_id")
            final_wh = overridden_entities.get("warehouse_id") or inherited_entities.get("warehouse_id")
            final_ch = overridden_entities.get("channel_id") or inherited_entities.get("channel_id")

            # Check if user query lacks explicit mentions and needs synthesized clarity
            if final_sku and not explicit_sku:
                supplements.append(f"SKU {final_sku}")
            if final_wh and not explicit_wh:
                supplements.append(f"warehouse {final_wh}")
            if final_ch and not explicit_channel:
                supplements.append(f"channel {final_ch}")

            if supplements and (_PRONOUN_CUES.search(sanitized_question) or current_domain is not None):
                # Replace pronouns or append entities
                replaced_q = _PRONOUN_CUES.sub("that", sanitized_question)
                resolved_question = f"{replaced_q} for {' and '.join(supplements)}"

        return ContextResolutionResult(
            original_question=question,
            resolved_question=resolved_question,
            inherited_entities=inherited_entities,
            overridden_entities=overridden_entities,
            inherited_filters=inherited_filters,
            overridden_filters=overridden_filters,
            incompatible_pruned=incompatible_pruned,
            corrections_applied=corrections_applied,
            time_shift_applied=time_shift_applied,
            ambiguity_resolved=ambiguity_resolved,
            inferred_domain=current_domain,
            lineage_events=lineage_events,
            confidence=ContextConfidence.HIGH if not sec_violations else ContextConfidence.LOW,
            sanitized_violations=sec_violations,
        )

    def _detect_correction(
        self,
        question: str,
        snapshot: ConversationContextSnapshot,
    ) -> Optional[CorrectionEvent]:
        """Detect explicit user corrections overriding existing context."""
        now_ts = datetime.now(timezone.utc).isoformat()
        corr_id = "CORR-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]

        # Warehouse correction
        wh_corr = re.search(r"\b(?:actually|correction|wait|no),?\s+(?:i\s+meant|not)\s+(?:warehouse\s+)?(wh[_-]\w+|wh\d+)\b", question, re.IGNORECASE)
        if wh_corr:
            new_wh = wh_corr.group(1).upper()
            return CorrectionEvent(
                correction_id=corr_id,
                turn_index=snapshot.current_turn_index,
                target_category=ContextCategory.ENTITY_CONTEXT,
                target_key="warehouse_id",
                previous_value=snapshot.entities.warehouse_id,
                new_value=new_wh,
                reason=f"User explicitly corrected warehouse to {new_wh}",
                timestamp=now_ts,
            )

        # SKU correction
        sku_corr = re.search(r"\b(?:actually|correction|wait|no),?\s+(?:i\s+meant|not)\s+(?:sku\s+)?(SKU[_-]?[A-Za-z0-9]+)\b", question, re.IGNORECASE)
        if sku_corr:
            new_sku = sku_corr.group(1).upper()
            return CorrectionEvent(
                correction_id=corr_id,
                turn_index=snapshot.current_turn_index,
                target_category=ContextCategory.ENTITY_CONTEXT,
                target_key="sku_id",
                previous_value=snapshot.entities.sku_id,
                new_value=new_sku,
                reason=f"User explicitly corrected SKU to {new_sku}",
                timestamp=now_ts,
            )

        # Filter removal
        forget_m = re.search(r"\b(?:forget|remove|clear|drop)\s+(?:the\s+)?(warehouse|channel|sku|supplier|filter)\b", question, re.IGNORECASE)
        if forget_m:
            target = forget_m.group(1).lower()
            key_map = {"warehouse": "warehouse_id", "channel": "channel_id", "sku": "sku_id", "supplier": "supplier_id"}
            attr_key = key_map.get(target, "warehouse_id")
            prev_val = getattr(snapshot.entities, attr_key, None)
            return CorrectionEvent(
                correction_id=corr_id,
                turn_index=snapshot.current_turn_index,
                target_category=ContextCategory.ENTITY_CONTEXT,
                target_key=attr_key,
                previous_value=prev_val,
                new_value=None,
                reason=f"User explicitly cleared {target} filter",
                timestamp=now_ts,
            )

        return None
