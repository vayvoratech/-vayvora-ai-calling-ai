"""Property Recommendation Service for Infra Builder Real Estate.

Provides tailored residential recommendations and suitability scoring based on
buyer requirements, preferred configuration, budget range, and location preference.
"""

from __future__ import annotations

from typing import Any, Dict


class PropertyRecommendationService:
    """Intelligent recommendation service mapping buyer requirements into suitable properties."""

    @staticmethod
    def get_recommendations(profile: Dict[str, Any]) -> Dict[str, Any]:
        interests = [str(i).lower() for i in profile.get("interest", [])]
        req_type = str(profile.get("property_type") or "").lower()
        budget = str(profile.get("budget") or "").lower()
        location = str(profile.get("location") or "").lower()

        is_villa = "villa" in req_type or any("villa" in i for i in interests) or "sarjapur" in location
        is_ready = "ready" in req_type or any("ready" in i for i in interests) or "electronic city" in location or "40" in budget or "50" in budget or "60" in budget

        if is_villa:
            return {
                "recommendedProperty": "Green Valley Villas",
                "location": "Sarjapur Road, Bangalore",
                "configurations": ["4 BHK (3,200 sq.ft)", "5 BHK (4,500 sq.ft)"],
                "priceRange": "₹2.5 Cr - ₹3.8 Cr",
                "matchScore": 95,
                "summary": "Luxury private villas with private gardens, swimming pool, and gated security.",
            }

        if is_ready:
            return {
                "recommendedProperty": "Urban Heights",
                "location": "Electronic City Phase 1, Bangalore",
                "configurations": ["1 BHK (650 sq.ft)", "2 BHK (1,050 sq.ft)"],
                "priceRange": "₹45 Lakhs - ₹68 Lakhs",
                "matchScore": 90,
                "summary": "Ready-to-move modern apartments adjacent to tech campuses and metro station.",
            }

        # Default recommendation: Skyline Towers
        return {
            "recommendedProperty": "Skyline Towers",
            "location": "Whitefield, Bangalore",
            "configurations": ["2 BHK (1,150 sq.ft)", "3 BHK (1,650 sq.ft)"],
            "priceRange": "₹85 Lakhs - ₹1.42 Cr",
            "matchScore": 92,
            "summary": "Premium high-rise apartments with 40+ lifestyle amenities, near ITPL and metro.",
        }


def get_property_recommendations(profile: Dict[str, Any]) -> Dict[str, Any]:
    return PropertyRecommendationService.get_recommendations(profile)


# Backward compatibility aliases
CourseRecommendationService = PropertyRecommendationService
get_course_recommendations = get_property_recommendations
