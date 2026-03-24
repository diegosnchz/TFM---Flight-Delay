"""
eu261.py - Logica de compensaciones bajo la normativa EU261/2004.

Este modulo encapsula las reglas del Reglamento (CE) n 261/2004 del
Parlamento Europeo: umbrales de retraso, bandas de distancia y calculo
de compensaciones economicas.

Referencia legal:
    Reglamento (CE) n 261/2004, articulos 5, 6 y 7.
    STJUE C-402/07 (Sturgeon): equipara cancelaciones con retrasos >= 3h.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import (
    EU261_COMPENSATION_BANDS,
    EU261_DELAY_THRESHOLD_MINUTES,
    setup_logging,
)

logger = setup_logging(__name__)


def get_eu261_compensation(distance_km: float) -> int:
    """
    Devuelve la compensacion EU261 aplicable segun la distancia de la ruta.

    Las bandas de compensacion segun el articulo 7 del Reglamento son:
      - Vuelos de hasta 1500 km: 250 EUR
      - Vuelos intracomunitarios de mas de 1500 km y
        vuelos de entre 1500 y 3500 km: 400 EUR
      - Todos los demas vuelos: 600 EUR

    Nota: El reglamento permite una reduccion del 50% si se ofrece
    transporte alternativo con llegada razonablemente cercana al horario
    original. Este modulo NO aplica esa reduccion (calcula el maximo).

    Args:
        distance_km: Distancia en kilometros entre origen y destino.

    Returns:
        Compensacion en euros (250, 400 o 600).

    Raises:
        ValueError: Si la distancia es negativa.

    Example:
        >>> get_eu261_compensation(1200)
        250
        >>> get_eu261_compensation(2000)
        400
        >>> get_eu261_compensation(5000)
        600
    """
    if distance_km < 0:
        raise ValueError(f"La distancia no puede ser negativa: {distance_km}")

    for min_km, max_km, compensation in EU261_COMPENSATION_BANDS:
        if min_km <= distance_km < max_km:
            return compensation

    # No deberia llegar aqui si EU261_COMPENSATION_BANDS esta bien definido
    logger.warning(
        "No se encontro banda de compensacion para distancia=%.1f km. "
        "Devolviendo 600 EUR por defecto.",
        distance_km,
    )
    return 600


def is_eu261_eligible(delay_minutes: float) -> bool:
    """
    Determina si un retraso es elegible para compensacion EU261.

    Segun el articulo 6 y la jurisprudencia Sturgeon (STJUE C-402/07),
    los pasajeros tienen derecho a compensacion cuando el retraso en la
    llegada supera las 3 horas (180 minutos).

    Args:
        delay_minutes: Retraso en minutos. Valores negativos indican
            adelanto (sin compensacion).

    Returns:
        True si el retraso es >= 180 minutos, False en caso contrario.

    Example:
        >>> is_eu261_eligible(200)
        True
        >>> is_eu261_eligible(150)
        False
        >>> is_eu261_eligible(-10)  # Vuelo adelantado
        False
    """
    return delay_minutes >= EU261_DELAY_THRESHOLD_MINUTES


def calculate_expected_value(
    p_delay: float,
    distance_km: float,
    ticket_price_eur: float,
    transport_cost_eur: float = 25.0,
    hourly_wage_eur: float = 8.87,
    hours_invested: float = 8.0,
    p_claim_denied: float = 0.30,
) -> dict[str, float]:
    """
    Calcula el Valor Esperado (EV) del modelo de arbitraje EU261 extendido.

    Formula:
        EV = P(retraso>=3h) * Compensacion * (1 - P(denegacion))
             - Precio_Billete
             - Coste_Transporte
             - Valor_Hora * Horas_Invertidas

    Este modelo asume que el "arbitrajista" compra el billete exclusivamente
    con la intencion de reclamar la compensacion si hay retraso. No considera
    el valor del desplazamiento en si mismo.

    Args:
        p_delay: Probabilidad predicha de retraso >= 3 horas (entre 0 y 1).
        distance_km: Distancia de la ruta en km (determina la compensacion).
        ticket_price_eur: Precio del billete de avion en EUR.
        transport_cost_eur: Coste estimado de transporte al aeropuerto
            (ida y vuelta). Por defecto 25 EUR.
        hourly_wage_eur: Valor del tiempo del arbitrajista en EUR/hora.
            Por defecto: salario minimo espanol 2025 (8.87 EUR/h).
        hours_invested: Horas totales invertidas en el viaje (desplazamiento
            al aeropuerto, espera, vuelo, gestion de la reclamacion).
            Por defecto 8 horas.
        p_claim_denied: Probabilidad de que la aerolinea deniegue la
            compensacion alegando circunstancias extraordinarias.
            Por defecto 0.30 (30%).

    Returns:
        Diccionario con los componentes del calculo:
            - "compensation_eur": Compensacion EU261 aplicable.
            - "expected_income": Ingreso esperado (p_delay * compensation * (1-p_denied)).
            - "total_cost": Coste total (billete + transporte + tiempo).
            - "ev": Valor esperado neto en EUR.
            - "profitable": True si EV > 0.

    Raises:
        ValueError: Si p_delay o p_claim_denied estan fuera de [0, 1].

    Example:
        >>> result = calculate_expected_value(0.08, 1200, 30.0)
        >>> result["ev"]
        -17.68
    """
    if not 0.0 <= p_delay <= 1.0:
        raise ValueError(f"p_delay debe estar entre 0 y 1: {p_delay}")
    if not 0.0 <= p_claim_denied <= 1.0:
        raise ValueError(f"p_claim_denied debe estar entre 0 y 1: {p_claim_denied}")

    compensation = get_eu261_compensation(distance_km)
    expected_income = p_delay * compensation * (1.0 - p_claim_denied)
    time_cost = hourly_wage_eur * hours_invested
    total_cost = ticket_price_eur + transport_cost_eur + time_cost
    ev = expected_income - total_cost

    return {
        "compensation_eur": float(compensation),
        "expected_income": round(expected_income, 2),
        "total_cost": round(total_cost, 2),
        "ev": round(ev, 2),
        "profitable": ev > 0.0,
    }


def breakeven_probability(
    distance_km: float,
    ticket_price_eur: float,
    transport_cost_eur: float = 25.0,
    hourly_wage_eur: float = 8.87,
    hours_invested: float = 8.0,
    p_claim_denied: float = 0.30,
) -> float:
    """
    Calcula la probabilidad minima de retraso para que el EV sea positivo.

    Despejando la formula del EV:
        p_min = total_cost / (compensation * (1 - p_claim_denied))

    Args:
        distance_km: Distancia de la ruta en km.
        ticket_price_eur: Precio del billete en EUR.
        transport_cost_eur: Coste de transporte al aeropuerto en EUR.
        hourly_wage_eur: Valor del tiempo en EUR/hora.
        hours_invested: Horas invertidas en el viaje.
        p_claim_denied: Probabilidad de reclamacion denegada.

    Returns:
        Probabilidad minima (entre 0 y 1) para que el arbitraje sea rentable.
        Si devuelve > 1, el arbitraje no es rentable bajo ninguna circunstancia.

    Example:
        >>> breakeven_probability(1200, 30.0)
        0.47  # aprox. - necesitaria un 47% de probabilidad de retraso
    """
    compensation = get_eu261_compensation(distance_km)
    time_cost = hourly_wage_eur * hours_invested
    total_cost = ticket_price_eur + transport_cost_eur + time_cost
    effective_compensation = compensation * (1.0 - p_claim_denied)

    if effective_compensation <= 0:
        return float("inf")

    p_min = total_cost / effective_compensation
    return round(p_min, 4)
