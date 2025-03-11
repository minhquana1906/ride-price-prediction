import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.decorators import timer

lat_min, lat_max = 20.98, 21.08
lon_min, lon_max = 105.75, 105.85
# Danh sách vùng nước (tọa độ trung tâm, bán kính)
water_areas = [
    {"name": "Ho Tay", "lat": 21.0580517, "lon": 105.82, "radius": 0.017},
    {"name": "Ho Guom", "lat": 21.0285, "lon": 105.852, "radius": 0.005},
    {"name": "Song Hong", "lat": 21.05, "lon": 105.85, "radius": 0.02},
]


def is_in_water(lat: float, lon: float) -> bool:
    """Check if a given location is in the water.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        bool: True if the location is in the water, False otherwise.
    """
    for area in water_areas:
        dist = np.sqrt((lat - area["lat"]) ** 2 + (lon - area["lon"]) ** 2)
        if dist < area["radius"]:
            return True
    return False


def apply_business_rules(data: pd.DataFrame) -> float:
    """
    Apply business rules and conditions to adjust the price of a ride.

    Args:
        data(pd.DataFrame): Single record of dataframe with data of a ride

    Returns:
        float: Adjusted price
    """
    base_price = data["base_price"]
    constrained_price = base_price

    # 1. Demand and supply
    demand = data["area_demand"]
    drivers = data["available_drivers"]
    demand_supply_ratio = demand / max(drivers, 1)

    if demand_supply_ratio > 2:
        surge_multiplier = min(1.5, 1 + (demand_supply_ratio - 2) * 0.1)
        constrained_price *= surge_multiplier

    # 2. Weather conditions
    weather = data["weather"]
    if weather == "rainy":
        constrained_price *= 1.2
    elif weather == "heavy_rain":
        constrained_price *= 1.5

    # 3. Traffic level
    traffic = data["traffic_level"]
    if traffic > 7:  # Heavy traffic
        constrained_price *= 1 + (traffic - 7) * 0.03

    # 4. Rush hours
    hour = data["hours"]
    if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19):
        constrained_price *= 1.25

    # 5. Discount for loyal customers
    previous_rides = data["user_previous_rides"]
    user_rating = data["user_rating"]

    if previous_rides > 50 and user_rating >= 4.5:
        constrained_price *= 0.90  # Discount 10% for users with more than 50 rides and rating >= 4.5
    elif previous_rides > 20:
        constrained_price *= 0.95  # Discount 5% for users with more than 20 rides

    # 6. Ensure the price is within the limits
    # Define min and max prices for each vehicle type
    min_price = 10000
    max_price = 10000000

    constrained_price = max(constrained_price, min_price)
    constrained_price = min(constrained_price, max_price)

    return np.round(constrained_price, -3)


def generate_synthetic_location_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic location data for model training and visualization purposes.

    Args:
        n_samples (int, optional): Number of locations. Defaults to 1000.
        seed (int, optional): Random seed for reusability and consistant recreation. Defaults to 42.

    Returns:
        pd.DataFrame: A dataframe with synthetic location data.
    """

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = [], [], [], []

    while len(pickup_lat) < n_samples:
        lat1, lon1 = np.random.uniform(lat_min, lat_max), np.random.uniform(lon_min, lon_max)
        lat2, lon2 = np.random.uniform(lat_min, lat_max), np.random.uniform(lon_min, lon_max)

        if not is_in_water(lat1, lon1) and not is_in_water(lat2, lon2):
            pickup_lat.append(lat1)
            pickup_lon.append(lon1)
            dropoff_lat.append(lat2)
            dropoff_lon.append(lon2)

    return pickup_lat, pickup_lon, dropoff_lat, dropoff_lon


@timer
def generate_synthetic_ride_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic ride data for model training and visualization purposes.

    Args:
        n_samples (int, optional): Number of rides. Defaults to 1000.
        seed (int, optional): Random seed for reusability and consistant recreation. Defaults to 42.

    Returns:
        pd.DataFrame: A dataframe with synthetic ride data.
    """

    np.random.seed(seed)
    random.seed(seed)

    # Create ride_id
    ride_id = [f"R{str(i).zfill(6)}" for i in range(n_samples)]

    # Timestamps take from last 30 days
    start_date = datetime.now() - timedelta(days=30)

    # define rush hours is 7-9am and 5-7pm
    preferred_hours = list(range(7, 10)) + list(range(17, 20))
    other_hours = [h for h in range(24) if h not in preferred_hours]

    # Weights: Prefer 7-9h and 17-19h more than other hours
    hour_weights = [3] * len(preferred_hours) + [1] * len(other_hours)
    all_hours = preferred_hours + other_hours

    # Gen data with weights
    booking_time = [
        start_date
        + timedelta(
            days=np.random.randint(0, 30),
            hours=int(np.random.choice(all_hours, p=np.array(hour_weights) / sum(hour_weights))),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60),
        )
        for _ in range(n_samples)
    ]

    hours = [bt.hour for bt in booking_time]
    days_of_week = [bt.weekday() for bt in booking_time]
    # is_weekend = [1 if d >= 5 else 0 for d in days_of_week]
    months = [bt.month for bt in booking_time]

    # Distance(km) and duration(min)
    # lognormal to ensure positive distance
    distance_km = np.random.lognormal(mean=1.5, sigma=0.5, size=n_samples)
    distance_km = np.round(distance_km, 1)

    avg_speed_kmh = 25
    duration_min = np.round(distance_km / avg_speed_kmh * 60 * (1 + np.random.lognormal(0, 0.2, n_samples)))

    # Weather conditions
    weather_conditions = np.random.choice(["sunny", "rainy", "heavy_rain"], size=n_samples, p=[0.5, 0.35, 0.15])

    # Traffic conditions
    base_traffic = np.random.normal(3, 1, n_samples)
    peak_hour_effect = np.array([3 if (7 <= h <= 9) or (17 <= h <= 19) else 0 for h in hours])
    traffic_level = np.clip(base_traffic + peak_hour_effect + np.random.normal(1, 0.5, n_samples), 0, 10)
    traffic_level = np.round(traffic_level).astype(int)

    # number of available drivers (less in rush hours)
    available_drivers = np.clip(np.random.normal(20, 5, n_samples) - peak_hour_effect / 2, 1, 50).astype(int)
    area_demand = np.clip(np.random.normal(50, 15, n_samples) + peak_hour_effect, 1, 100).astype(int)

    # Vehicle types
    vehicle_types = np.random.choice(
        ["motorbike", "4_seater_car", "7_seater_car", "luxury_car"],
        size=n_samples,
        p=[0.45, 0.35, 0.15, 0.05],
    )

    # User information
    # User rating to consider for sales
    user_rating = np.clip(np.random.normal(4.5, 0.5, n_samples), 1, 5)
    user_rating = np.round(user_rating, 1)

    # User's previous rides
    user_previous_rides = np.clip(np.random.exponential(scale=20, size=n_samples), 0, 500).astype(int)

    # define the base price per km for each vehicle type
    base_price_per_km = {
        "motorbike": 10000,
        "4_seater_car": 20000,
        "7_seater_car": 30000,
        "luxury_car": 60000,
    }

    # Calculate fare based on distance and vehicle type, rounding to the nearest 1000
    base_price = np.round(np.array([base_price_per_km[vt] for vt in vehicle_types]) * distance_km, -3)

    # TODO: Add location(piclup, dropoff) data
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = generate_synthetic_location_data(n_samples)

    # Create a dataframe
    df = pd.DataFrame({
        "ride_id": ride_id,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "booking_time": booking_time,
        "hours": hours,
        "day_of_week": days_of_week,
        "months": months,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "weather": weather_conditions,
        "traffic_level": traffic_level,
        "available_drivers": available_drivers,
        "area_demand": area_demand,
        "vehicle_types": vehicle_types,
        "user_rating": user_rating,
        "user_previous_rides": user_previous_rides,
        "base_price": base_price,
    })

    df["constrained_price"] = df.apply(apply_business_rules, axis=1)

    return df


if __name__ == "__main__":
    df = generate_synthetic_ride_data(n_samples=10000)
    df.to_csv("dataset/synthetic_ride_data.csv", index=False)
