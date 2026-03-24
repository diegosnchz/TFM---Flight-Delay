"""
test_eu261.py - Tests unitarios para la logica de compensaciones EU261.

Ejecutar con: python -m pytest tests/ -v
"""

import pytest

from src.utils.eu261 import (
    breakeven_probability,
    calculate_expected_value,
    get_eu261_compensation,
    is_eu261_eligible,
)


class TestGetEu261Compensation:
    """Tests para la funcion de calculo de compensacion por distancia."""

    def test_short_route_250(self):
        """Rutas de menos de 1500 km -> 250 EUR."""
        assert get_eu261_compensation(0) == 250
        assert get_eu261_compensation(800) == 250
        assert get_eu261_compensation(1499) == 250

    def test_medium_route_400(self):
        """Rutas de 1500 a 3500 km -> 400 EUR."""
        assert get_eu261_compensation(1500) == 400
        assert get_eu261_compensation(2000) == 400
        assert get_eu261_compensation(3499) == 400

    def test_long_route_600(self):
        """Rutas de mas de 3500 km -> 600 EUR."""
        assert get_eu261_compensation(3500) == 600
        assert get_eu261_compensation(5000) == 600
        assert get_eu261_compensation(10000) == 600

    def test_negative_distance_raises(self):
        """Una distancia negativa debe lanzar ValueError."""
        with pytest.raises(ValueError):
            get_eu261_compensation(-1)


class TestIsEu261Eligible:
    """Tests para la funcion de elegibilidad EU261."""

    def test_eligible_at_180_minutes(self):
        """Exactamente 180 minutos -> elegible."""
        assert is_eu261_eligible(180) is True

    def test_eligible_above_threshold(self):
        """Por encima del umbral -> elegible."""
        assert is_eu261_eligible(181) is True
        assert is_eu261_eligible(600) is True

    def test_not_eligible_below_threshold(self):
        """Por debajo del umbral -> no elegible."""
        assert is_eu261_eligible(179) is False
        assert is_eu261_eligible(0) is False

    def test_early_flight_not_eligible(self):
        """Vuelo adelantado -> no elegible."""
        assert is_eu261_eligible(-30) is False


class TestCalculateExpectedValue:
    """Tests para el modelo de Expected Value."""

    def test_ev_structure(self):
        """El resultado debe tener todas las claves esperadas."""
        result = calculate_expected_value(
            p_delay=0.10,
            distance_km=1200,
            ticket_price_eur=30.0,
        )
        assert "compensation_eur" in result
        assert "expected_income" in result
        assert "total_cost" in result
        assert "ev" in result
        assert "profitable" in result

    def test_high_probability_profitable(self):
        """Con alta probabilidad de retraso y billete barato, EV debe ser positivo."""
        result = calculate_expected_value(
            p_delay=0.90,
            distance_km=1200,
            ticket_price_eur=10.0,
            transport_cost_eur=10.0,
            hourly_wage_eur=5.0,
            hours_invested=3.0,
            p_claim_denied=0.10,
        )
        assert result["ev"] > 0
        assert result["profitable"] is True

    def test_low_probability_not_profitable(self):
        """Con baja probabilidad de retraso y billete caro, EV debe ser negativo."""
        result = calculate_expected_value(
            p_delay=0.01,
            distance_km=1200,
            ticket_price_eur=80.0,
        )
        assert result["ev"] < 0
        assert result["profitable"] is False

    def test_compensation_matches_distance(self):
        """La compensacion debe corresponder a la banda de distancia correcta."""
        result_short = calculate_expected_value(0.10, 800, 30.0)
        result_medium = calculate_expected_value(0.10, 2000, 30.0)
        result_long = calculate_expected_value(0.10, 5000, 30.0)

        assert result_short["compensation_eur"] == 250
        assert result_medium["compensation_eur"] == 400
        assert result_long["compensation_eur"] == 600

    def test_invalid_p_delay_raises(self):
        """p_delay fuera de [0, 1] debe lanzar ValueError."""
        with pytest.raises(ValueError):
            calculate_expected_value(p_delay=1.5, distance_km=1000, ticket_price_eur=30)

        with pytest.raises(ValueError):
            calculate_expected_value(p_delay=-0.1, distance_km=1000, ticket_price_eur=30)


class TestBreakevenProbability:
    """Tests para el calculo de la probabilidad de break-even."""

    def test_breakeven_is_positive(self):
        """La probabilidad de break-even siempre debe ser positiva."""
        p_min = breakeven_probability(1200, 30.0)
        assert p_min > 0

    def test_higher_ticket_price_increases_breakeven(self):
        """A mayor precio del billete, se necesita mayor probabilidad de retraso."""
        p_cheap = breakeven_probability(1200, 10.0)
        p_expensive = breakeven_probability(1200, 80.0)
        assert p_expensive > p_cheap

    def test_higher_compensation_decreases_breakeven(self):
        """A mayor compensacion (ruta mas larga), el break-even es mas facil de alcanzar."""
        p_short = breakeven_probability(800, 30.0)   # 250 EUR compensacion
        p_long = breakeven_probability(5000, 30.0)   # 600 EUR compensacion
        assert p_short > p_long
