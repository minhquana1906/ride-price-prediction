class PricingSystem:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.price_constraints = {
            "min_multiplier": 0.8,  # 20% discount
            "max_multiplier": 2.0,  # 20% surge
            "min_price": {
                "motorbike": 10000,
                "4_seater_car": 15000,
                "7_seater_car": 20000,
                "luxury_car": 35000,
            },
        }
