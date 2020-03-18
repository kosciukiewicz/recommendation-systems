from abc import ABC, abstractmethod

class RecommendationMethod(ABC):

    @abstractmethod
    def fit(self, user_items_ratings):
        pass

    @abstractmethod
    def get_recommendations(self, user_id, n):
        pass
