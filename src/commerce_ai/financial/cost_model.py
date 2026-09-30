"""Cost Component Model & Availability Resolver (Phase 6B).

Provides deterministic cost component modeling, source traceability, and availability auditing for:
- PRODUCT_COST
- SHIPPING_COST
- PAYMENT_PROCESSING_COST
- PACKAGING_COST
- WAREHOUSE_HANDLING_COST
- RETURN_PROCESSING_COST
- OTHER_VARIABLE_COST

Explicitly distinguishes:
- SOURCE_DATA: Observed directly in transaction records
- CATALOG_ESTIMATE: Derived from catalog/procurement master tables (e.g. products.csv)
- CONFIGURED_ASSUMPTION: Applied from explicit parametric business defaults
- UNAVAILABLE: Unobserved cost components (never silently assumed as zero)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.schemas import (
    ContributionMarginStatus,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostModelConfig,
    CostSourceType,
    UnitEconomicsStatus,
)


class CostModelResolver:
    """Resolves cost components and source traceability for transactions."""

    def __init__(self, config: Optional[CostModelConfig] = None):
        self.config = config or CostModelConfig()

    def resolve_line_costs(
        self,
        quantity: Optional[int],
        unit_price: Optional[float],
        net_revenue: Optional[float],
        sku_id: str,
        channel_id: str,
        warehouse_id: str,
        unit_cost: Optional[float] = None,
        transaction_costs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, CostComponentDetail]:
        """Resolve all 7 cost components for an individual sales transaction line.

        Returns a dictionary mapping each CostComponent name to its CostComponentDetail.
        """
        txn = transaction_costs or {}
        details: Dict[str, CostComponentDetail] = {}

        # 1. PRODUCT_COST
        prod_cost_amount: Optional[float] = None
        prod_unit_cost: Optional[float] = None
        prod_status = CostComponentStatus.UNAVAILABLE
        prod_src = CostSourceType.UNAVAILABLE
        prod_ref = None

        if txn.get("product_cost") is not None:
            prod_cost_amount = float(txn["product_cost"])
            prod_unit_cost = prod_cost_amount / quantity if quantity and quantity > 0 else None
            prod_status = CostComponentStatus.AVAILABLE_SOURCE
            prod_src = CostSourceType.SOURCE_DATA
            prod_ref = "sales.product_cost"
        elif unit_cost is not None and not np.isnan(unit_cost) and unit_cost >= 0:
            prod_unit_cost = float(unit_cost)
            prod_cost_amount = float(quantity * unit_cost) if quantity is not None else None
            prod_status = CostComponentStatus.AVAILABLE_ESTIMATED
            prod_src = CostSourceType.CATALOG_ESTIMATE
            prod_ref = "products.csv:unit_cost"

        details[CostComponent.PRODUCT_COST.value] = CostComponentDetail(
            component=CostComponent.PRODUCT_COST,
            amount=prod_cost_amount,
            unit_amount=prod_unit_cost,
            status=prod_status,
            source_type=prod_src,
            source_reference=prod_ref,
            description="Catalog-based procurement cost estimate" if prod_src == CostSourceType.CATALOG_ESTIMATE else "Transactional product cost",
        )

        # 2. SHIPPING_COST
        ship_amount: Optional[float] = None
        ship_unit: Optional[float] = None
        ship_status = CostComponentStatus.UNAVAILABLE
        ship_src = CostSourceType.UNAVAILABLE
        ship_ref = None

        if txn.get("shipping_cost") is not None:
            ship_amount = float(txn["shipping_cost"])
            ship_unit = ship_amount / quantity if quantity and quantity > 0 else None
            ship_status = CostComponentStatus.AVAILABLE_SOURCE
            ship_src = CostSourceType.SOURCE_DATA
            ship_ref = "sales.shipping_cost"
        elif channel_id in self.config.channel_shipping_costs:
            rate = self.config.channel_shipping_costs[channel_id]
            ship_unit = rate
            ship_amount = rate * quantity if quantity is not None else None
            ship_status = CostComponentStatus.AVAILABLE_ASSUMED
            ship_src = CostSourceType.CONFIGURED_ASSUMPTION
            ship_ref = f"config.channel_shipping_costs[{channel_id}]"
        elif self.config.default_shipping_cost_per_unit is not None:
            rate = self.config.default_shipping_cost_per_unit
            ship_unit = rate
            ship_amount = rate * quantity if quantity is not None else None
            ship_status = CostComponentStatus.AVAILABLE_ASSUMED
            ship_src = CostSourceType.CONFIGURED_ASSUMPTION
            ship_ref = "config.default_shipping_cost_per_unit"
        elif self.config.default_shipping_cost_per_order is not None:
            flat = self.config.default_shipping_cost_per_order
            ship_amount = flat
            ship_unit = flat / quantity if quantity and quantity > 0 else flat
            ship_status = CostComponentStatus.AVAILABLE_ASSUMED
            ship_src = CostSourceType.CONFIGURED_ASSUMPTION
            ship_ref = "config.default_shipping_cost_per_order"

        details[CostComponent.SHIPPING_COST.value] = CostComponentDetail(
            component=CostComponent.SHIPPING_COST,
            amount=ship_amount,
            unit_amount=ship_unit,
            status=ship_status,
            source_type=ship_src,
            source_reference=ship_ref,
            description="Outbound fulfillment freight cost",
        )

        # 3. PAYMENT_PROCESSING_COST
        pay_amount: Optional[float] = None
        pay_unit: Optional[float] = None
        pay_status = CostComponentStatus.UNAVAILABLE
        pay_src = CostSourceType.UNAVAILABLE
        pay_ref = None

        raw_pay = txn.get("payment_processing_cost", txn.get("payment_fee"))
        if raw_pay is not None:
            pay_amount = float(raw_pay)
            pay_unit = pay_amount / quantity if quantity and quantity > 0 else None
            pay_status = CostComponentStatus.AVAILABLE_SOURCE
            pay_src = CostSourceType.SOURCE_DATA
            pay_ref = "sales.payment_processing_cost"
        else:
            pay_rate = self.config.channel_payment_processing_rates.get(
                channel_id, self.config.default_payment_processing_rate
            )
            pay_fixed = self.config.channel_payment_processing_fixed_fees.get(
                channel_id, self.config.default_payment_processing_fixed_fee
            )
            if pay_rate is not None or pay_fixed is not None:
                fee = 0.0
                if pay_rate is not None and net_revenue is not None and net_revenue > 0:
                    fee += pay_rate * net_revenue
                if pay_fixed is not None:
                    fee += pay_fixed
                pay_amount = float(fee)
                pay_unit = pay_amount / quantity if quantity and quantity > 0 else None
                pay_status = CostComponentStatus.AVAILABLE_ASSUMED
                pay_src = CostSourceType.CONFIGURED_ASSUMPTION
                pay_ref = f"config.channel_payment_processing_rates[{channel_id}]"

        details[CostComponent.PAYMENT_PROCESSING_COST.value] = CostComponentDetail(
            component=CostComponent.PAYMENT_PROCESSING_COST,
            amount=pay_amount,
            unit_amount=pay_unit,
            status=pay_status,
            source_type=pay_src,
            source_reference=pay_ref,
            description="Payment gateway transaction fee",
        )

        # 4. PACKAGING_COST
        pkg_amount: Optional[float] = None
        pkg_unit: Optional[float] = None
        pkg_status = CostComponentStatus.UNAVAILABLE
        pkg_src = CostSourceType.UNAVAILABLE
        pkg_ref = None

        if txn.get("packaging_cost") is not None:
            pkg_amount = float(txn["packaging_cost"])
            pkg_unit = pkg_amount / quantity if quantity and quantity > 0 else None
            pkg_status = CostComponentStatus.AVAILABLE_SOURCE
            pkg_src = CostSourceType.SOURCE_DATA
            pkg_ref = "sales.packaging_cost"
        elif self.config.default_packaging_cost_per_unit is not None:
            rate = self.config.default_packaging_cost_per_unit
            pkg_unit = rate
            pkg_amount = rate * quantity if quantity is not None else None
            pkg_status = CostComponentStatus.AVAILABLE_ASSUMED
            pkg_src = CostSourceType.CONFIGURED_ASSUMPTION
            pkg_ref = "config.default_packaging_cost_per_unit"

        details[CostComponent.PACKAGING_COST.value] = CostComponentDetail(
            component=CostComponent.PACKAGING_COST,
            amount=pkg_amount,
            unit_amount=pkg_unit,
            status=pkg_status,
            source_type=pkg_src,
            source_reference=pkg_ref,
            description="Packaging materials and dunnage cost",
        )

        # 5. WAREHOUSE_HANDLING_COST
        wh_amount: Optional[float] = None
        wh_unit: Optional[float] = None
        wh_status = CostComponentStatus.UNAVAILABLE
        wh_src = CostSourceType.UNAVAILABLE
        wh_ref = None

        raw_wh = txn.get("warehouse_handling_cost", txn.get("handling_cost"))
        if raw_wh is not None:
            wh_amount = float(raw_wh)
            wh_unit = wh_amount / quantity if quantity and quantity > 0 else None
            wh_status = CostComponentStatus.AVAILABLE_SOURCE
            wh_src = CostSourceType.SOURCE_DATA
            wh_ref = "sales.warehouse_handling_cost"
        elif warehouse_id in self.config.warehouse_handling_cost_per_unit:
            rate = self.config.warehouse_handling_cost_per_unit[warehouse_id]
            wh_unit = rate
            wh_amount = rate * quantity if quantity is not None else None
            wh_status = CostComponentStatus.AVAILABLE_ASSUMED
            wh_src = CostSourceType.CONFIGURED_ASSUMPTION
            wh_ref = f"config.warehouse_handling_cost_per_unit[{warehouse_id}]"
        elif self.config.default_warehouse_handling_cost_per_unit is not None:
            rate = self.config.default_warehouse_handling_cost_per_unit
            wh_unit = rate
            wh_amount = rate * quantity if quantity is not None else None
            wh_status = CostComponentStatus.AVAILABLE_ASSUMED
            wh_src = CostSourceType.CONFIGURED_ASSUMPTION
            wh_ref = "config.default_warehouse_handling_cost_per_unit"

        details[CostComponent.WAREHOUSE_HANDLING_COST.value] = CostComponentDetail(
            component=CostComponent.WAREHOUSE_HANDLING_COST,
            amount=wh_amount,
            unit_amount=wh_unit,
            status=wh_status,
            source_type=wh_src,
            source_reference=wh_ref,
            description="Fulfillment node pick, pack, and handling labor cost",
        )

        # 6. RETURN_PROCESSING_COST
        ret_amount: Optional[float] = None
        ret_unit: Optional[float] = None
        ret_status = CostComponentStatus.UNAVAILABLE
        ret_src = CostSourceType.UNAVAILABLE
        ret_ref = None

        if txn.get("return_processing_cost") is not None:
            ret_amount = float(txn["return_processing_cost"])
            ret_unit = ret_amount / quantity if quantity and quantity > 0 else None
            ret_status = CostComponentStatus.AVAILABLE_SOURCE
            ret_src = CostSourceType.SOURCE_DATA
            ret_ref = "sales.return_processing_cost"
        elif self.config.default_return_processing_cost_per_unit is not None:
            rate = self.config.default_return_processing_cost_per_unit
            ret_unit = rate
            ret_amount = rate * quantity if quantity is not None else None
            ret_status = CostComponentStatus.AVAILABLE_ASSUMED
            ret_src = CostSourceType.CONFIGURED_ASSUMPTION
            ret_ref = "config.default_return_processing_cost_per_unit"

        details[CostComponent.RETURN_PROCESSING_COST.value] = CostComponentDetail(
            component=CostComponent.RETURN_PROCESSING_COST,
            amount=ret_amount,
            unit_amount=ret_unit,
            status=ret_status,
            source_type=ret_src,
            source_reference=ret_ref,
            description="Reverse logistics, customer return, and restocking fee",
        )

        # 7. OTHER_VARIABLE_COST
        oth_amount: Optional[float] = None
        oth_unit: Optional[float] = None
        oth_status = CostComponentStatus.UNAVAILABLE
        oth_src = CostSourceType.UNAVAILABLE
        oth_ref = None

        if txn.get("other_variable_cost") is not None:
            oth_amount = float(txn["other_variable_cost"])
            oth_unit = oth_amount / quantity if quantity and quantity > 0 else None
            oth_status = CostComponentStatus.AVAILABLE_SOURCE
            oth_src = CostSourceType.SOURCE_DATA
            oth_ref = "sales.other_variable_cost"
        elif self.config.default_other_variable_cost_per_unit is not None:
            rate = self.config.default_other_variable_cost_per_unit
            oth_unit = rate
            oth_amount = rate * quantity if quantity is not None else None
            oth_status = CostComponentStatus.AVAILABLE_ASSUMED
            oth_src = CostSourceType.CONFIGURED_ASSUMPTION
            oth_ref = "config.default_other_variable_cost_per_unit"

        details[CostComponent.OTHER_VARIABLE_COST.value] = CostComponentDetail(
            component=CostComponent.OTHER_VARIABLE_COST,
            amount=oth_amount,
            unit_amount=oth_unit,
            status=oth_status,
            source_type=oth_src,
            source_reference=oth_ref,
            description="Miscellaneous variable commercial fees",
        )

        return details

    def calculate_cost_completeness(
        self,
        cost_details: Dict[str, CostComponentDetail],
        required_components: Optional[List[CostComponent]] = None,
    ) -> Tuple[float, List[str], List[str], List[str], List[str]]:
        """Calculate deterministic cost completeness and partition components by status.

        Returns:
            Tuple of:
            - completeness_pct (float 0.0 to 100.0)
            - available_components (List[str])
            - missing_components (List[str])
            - estimated_components (List[str])
            - assumed_components (List[str])
        """
        req_list = required_components or self.config.required_cost_components
        req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in req_list]

        available: List[str] = []
        missing: List[str] = []
        estimated: List[str] = []
        assumed: List[str] = []

        valid_statuses = {
            CostComponentStatus.AVAILABLE_SOURCE,
            CostComponentStatus.AVAILABLE_ESTIMATED,
            CostComponentStatus.AVAILABLE_ASSUMED,
        }

        for comp_name, detail in cost_details.items():
            if detail.status in valid_statuses:
                available.append(comp_name)
                if detail.source_type == CostSourceType.CATALOG_ESTIMATE:
                    estimated.append(comp_name)
                elif detail.source_type == CostSourceType.CONFIGURED_ASSUMPTION:
                    assumed.append(comp_name)
            else:
                missing.append(comp_name)

        # Completeness is calculated relative to required cost components
        req_available_count = sum(1 for k in req_keys if k in available)
        completeness_pct = (req_available_count / len(req_keys)) * 100.0 if req_keys else 100.0

        return completeness_pct, available, missing, estimated, assumed

    def get_cost_model_summary(self) -> Dict[str, Any]:
        """Produce an audit summary of the configured cost model and availability."""
        req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in self.config.required_cost_components]
        return {
            "required_cost_components": req_keys,
            "assumptions_configured": {
                "channel_shipping_costs": self.config.channel_shipping_costs,
                "default_shipping_cost_per_unit": self.config.default_shipping_cost_per_unit,
                "default_shipping_cost_per_order": self.config.default_shipping_cost_per_order,
                "channel_payment_processing_rates": self.config.channel_payment_processing_rates,
                "channel_payment_processing_fixed_fees": self.config.channel_payment_processing_fixed_fees,
                "default_packaging_cost_per_unit": self.config.default_packaging_cost_per_unit,
                "warehouse_handling_cost_per_unit": self.config.warehouse_handling_cost_per_unit,
                "default_return_processing_cost_per_unit": self.config.default_return_processing_cost_per_unit,
                "default_other_variable_cost_per_unit": self.config.default_other_variable_cost_per_unit,
            },
            "canonical_status": {
                CostComponent.PRODUCT_COST.value: CostComponentStatus.AVAILABLE_ESTIMATED.value,
                CostComponent.SHIPPING_COST.value: CostComponentStatus.UNAVAILABLE.value,
                CostComponent.PAYMENT_PROCESSING_COST.value: CostComponentStatus.UNAVAILABLE.value,
                CostComponent.PACKAGING_COST.value: CostComponentStatus.UNAVAILABLE.value,
                CostComponent.WAREHOUSE_HANDLING_COST.value: CostComponentStatus.UNAVAILABLE.value,
                CostComponent.RETURN_PROCESSING_COST.value: CostComponentStatus.UNAVAILABLE.value,
                CostComponent.OTHER_VARIABLE_COST.value: CostComponentStatus.UNAVAILABLE.value,
            },
            "notice": (
                "The current canonical dataset contains only product procurement cost (PRODUCT_COST) "
                "from the catalog master. All other variable cost components are UNAVAILABLE unless "
                "injected via transaction fields or configured assumptions."
            ),
        }
