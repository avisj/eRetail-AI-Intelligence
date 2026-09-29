"""Tests for Synthetic Data Generator."""

from commerce_ai.data.generators import (
    GeneratorConfig,
    SyntheticDataGenerator,
    generate_sample_dataset,
)
from commerce_ai.data.validators import CommerceDataValidator


class TestSyntheticDataGenerator:
    def test_generator_creates_all_eight_entities(self):
        config = GeneratorConfig(
            number_of_skus=15,
            number_of_warehouses=2,
            number_of_channels=3,
            number_of_suppliers=5,
            historical_days=30,
            random_seed=42,
        )
        generator = SyntheticDataGenerator(config)
        datasets = generator.generate_all()

        expected_entities = [
            "channels", "warehouses", "suppliers", "products",
            "purchases", "sales", "returns", "inventory",
        ]
        for ent in expected_entities:
            assert ent in datasets, f"Missing entity {ent}"
            assert len(datasets[ent]) > 0, f"Entity {ent} is empty"

    def test_referential_integrity_of_generated_data(self):
        config = GeneratorConfig(
            number_of_skus=20,
            number_of_warehouses=3,
            number_of_channels=3,
            number_of_suppliers=5,
            historical_days=45,
            random_seed=123,
        )
        datasets = generate_sample_dataset(
            number_of_skus=config.number_of_skus,
            number_of_warehouses=config.number_of_warehouses,
            number_of_channels=config.number_of_channels,
            number_of_suppliers=config.number_of_suppliers,
            historical_days=config.historical_days,
            random_seed=config.random_seed,
        )

        validator = CommerceDataValidator()
        res = validator.validate_all(datasets)
        assert res.valid is True, f"Validation errors: {[e.issue for e in res.errors]}"

    def test_generator_reproducibility(self):
        config1 = GeneratorConfig(number_of_skus=10, number_of_warehouses=2, historical_days=20, random_seed=999)
        config2 = GeneratorConfig(number_of_skus=10, number_of_warehouses=2, historical_days=20, random_seed=999)

        gen1 = SyntheticDataGenerator(config1).generate_all()
        gen2 = SyntheticDataGenerator(config2).generate_all()

        assert gen1["sales"].equals(gen2["sales"])
        assert gen1["inventory"].equals(gen2["inventory"])
        assert gen1["products"].equals(gen2["products"])

    def test_controlled_anomalies_injection(self):
        config = GeneratorConfig(
            number_of_skus=10,
            number_of_warehouses=2,
            historical_days=15,
            random_seed=42,
            include_controlled_anomalies=True,
        )
        datasets = SyntheticDataGenerator(config).generate_all()
        validator = CommerceDataValidator()
        res = validator.validate_all(datasets)
        # Should have caught the injected anomaly!
        assert res.valid is False
        assert len(res.errors) > 0
