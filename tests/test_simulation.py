from simulation.scenario_engine import HealthRiskSimulationEngine


def test_simulation_runs():
    engine = HealthRiskSimulationEngine(starting_capital=1_000_000)

    history = engine.run_simulation(years=1)
    score = engine.final_score()

    assert len(history) == 4
    assert "total_score" in score
    assert "certification_tier" in score
    assert score["quarters_completed"] == 4


def test_scenario_catalog_has_ten_scenarios():
    engine = HealthRiskSimulationEngine()

    catalog = engine.scenario_catalog()

    assert len(catalog) >= 10