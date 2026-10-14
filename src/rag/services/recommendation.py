"""JB Infra recommendation service for the demo MVP.

Provides simple project recommendations based on customer preferences.
This is intentionally lightweight and deterministic for the MVP.
"""

from __future__ import annotations

from typing import Any, Dict, List


PROJECTS = {
    "pristine_city": {
        "name": "Pristine City",
        "keywords": [
            "vikarabad",
            "pristine",
            "dtcp",
            "rera",
            "150 acres",
            "rainwater harvesting",
        ],
    },
    "nature_valley": {
        "name": "Nature Valley",
        "keywords": [
            "nature valley",
            "700 acres",
            "cycling",
            "green zone",
            "6000 plots",
        ],
    },
    "serene_city": {
        "name": "Serene City",
        "keywords": [
            "ibrahimpatnam",
            "serene city",
            "hmda",
            "villa",
            "commercial",
            "clubhouse",
        ],
    },
    "jb_pride": {
        "name": "JB Pride",
        "keywords": [
            "tandur",
            "mominpet",
            "jb pride",
            "sports",
            "cricket",
            "amphitheatre",
            "herbal garden",
        ],
    },
}


class ProjectRecommendationService:
    """Recommend JB Infra projects from customer preferences."""

    @staticmethod
    def get_recommendations(
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a small ranked list of relevant projects."""

        interests = profile.get(
            "interest",
            profile.get("interests", []),
        )

        if isinstance(interests, str):
            interests = [interests]

        location = str(
            profile.get("location", "")
        ).lower()

        budget = str(
            profile.get("budget", "")
        ).lower()

        purpose = str(
            profile.get("purpose", "")
        ).lower()

        preferences = " ".join(
            [
                str(item).lower()
                for item in interests
            ]
        )

        preferences = " ".join(
            [
                preferences,
                location,
                budget,
                purpose,
            ]
        )

        scored_projects: List[Dict[str, Any]] = []

        for project_id, project in PROJECTS.items():
            score = 0
            matched_keywords: List[str] = []

            for keyword in project["keywords"]:
                if keyword.lower() in preferences:
                    score += 1
                    matched_keywords.append(keyword)

            scored_projects.append(
                {
                    "projectId": project_id,
                    "projectName": project["name"],
                    "score": score,
                    "matchedKeywords": matched_keywords,
                }
            )

        scored_projects.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        recommended = [
            project
            for project in scored_projects
            if project["score"] > 0
        ][:3]

        # If the customer has not supplied enough information,
        # return the portfolio rather than inventing a preference.
        if not recommended:
            recommended = [
                {
                    "projectId": project_id,
                    "projectName": project["name"],
                    "score": 0,
                    "matchedKeywords": [],
                }
                for project_id, project in list(
                    PROJECTS.items()
                )[:3]
            ]

        return {
            "recommendedProjects": recommended,
            "recommendationCount": len(recommended),
            "profile": profile,
            "note": (
                "Recommendations are based only on the supplied "
                "customer preferences and the JB Infra demo "
                "knowledge base."
            ),
        }


def get_project_recommendations(
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Compatibility wrapper for project recommendations."""

    return ProjectRecommendationService.get_recommendations(
        profile
    )


# Backward-compatible aliases for existing imports.
CourseRecommendationService = ProjectRecommendationService


def get_course_recommendations(
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Compatibility wrapper retained for existing imports."""

    return get_project_recommendations(profile)


__all__ = [
    "PROJECTS",
    "ProjectRecommendationService",
    "get_project_recommendations",
    "CourseRecommendationService",
    "get_course_recommendations",
]