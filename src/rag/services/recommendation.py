"""Course Recommendation Service for EduSaaS and Vayvora Tech.

Provides tailored curricula recommendations and readiness scoring based on
learner background, academic year, career goals, and technical domain interests.
"""

from __future__ import annotations

from typing import Any, Dict


class CourseRecommendationService:
    """Intelligent recommendation service mapping learner profile attributes into tailored curricula."""

    @staticmethod
    def get_recommendations(profile: Dict[str, Any]) -> Dict[str, Any]:
        interests = [str(i).lower() for i in profile.get("interest", [])]
        year = str(profile.get("year") or "").lower()
        experience = str(profile.get("experience") or "beginner").lower()

        has_ai = any("ai" in i or "machine learning" in i or "ml" in i for i in interests)
        has_fs = any("full" in i or "stack" in i or "web" in i for i in interests)
        has_ds = any("data" in i or "analytics" in i or "sql" in i for i in interests)
        has_cloud = any("cloud" in i or "devops" in i or "aws" in i for i in interests)
        has_cyber = any("cyber" in i or "security" in i or "hacking" in i for i in interests)
        has_dsa = any("dsa" in i or "algorithm" in i or "system design" in i for i in interests)

        # Readiness score
        readiness_score = 40
        if "3rd" in year or "third" in year:
            readiness_score += 2
        elif "4th" in year or "final" in year:
            readiness_score += 10
        elif "2nd" in year or "second" in year:
            readiness_score -= 2

        if experience in ("intermediate", "moderate"):
            readiness_score += 15
        elif experience in ("advanced", "experienced"):
            readiness_score += 25

        if has_ai and has_fs:
            return {
                "recommendedPath": "AI + Full Stack",
                "courses": [
                    "Python",
                    "Machine Learning",
                    "Generative AI",
                    "Java/Spring Boot",
                    "React",
                ],
                "readinessScore": readiness_score,
                "summary": (
                    "A dual-track combining Full-Stack Web Development fundamentals "
                    "with Generative AI and intelligent agent architectures."
                ),
            }

        if has_fs:
            return {
                "recommendedPath": "Full-Stack Web Development",
                "courses": [
                    "React 19 & Next.js 15",
                    "Node.js & TypeScript",
                    "FastAPI & PostgreSQL",
                    "Docker & Cloud Deployment",
                    "Full-Stack Production Capstone",
                ],
                "readinessScore": readiness_score + 10,
                "summary": "End-to-end modern web and distributed applications engineering.",
            }

        if has_ai:
            return {
                "recommendedPath": "AI & Machine Learning",
                "courses": [
                    "Python & Numerical Computing",
                    "Supervised & Unsupervised Machine Learning",
                    "Deep Learning & PyTorch",
                    "Generative AI & LLMs",
                    "Production AI Agent Capstones",
                ],
                "readinessScore": readiness_score + 8,
                "summary": "Deep foundations in modern predictive models and generative AI systems.",
            }

        if has_ds:
            return {
                "recommendedPath": "Data Science & Analytics",
                "courses": [
                    "Python & Pandas",
                    "Advanced SQL",
                    "Tableau & Power BI",
                    "Predictive Modeling",
                    "Financial Fraud Detection Capstone",
                ],
                "readinessScore": readiness_score + 6,
                "summary": "Data extraction, business intelligence, and predictive statistical analytics.",
            }

        if has_cloud:
            return {
                "recommendedPath": "Cloud & DevOps Engineering",
                "courses": [
                    "AWS Core Cloud Architecture",
                    "Docker Containerization",
                    "Kubernetes Orchestration",
                    "Terraform Infrastructure as Code",
                    "CI/CD Production Deployment Capstone",
                ],
                "readinessScore": readiness_score + 12,
                "summary": "Automated cloud infrastructure, container orchestration, and reliability engineering.",
            }

        if has_cyber:
            return {
                "recommendedPath": "Cybersecurity & Ethical Hacking",
                "courses": [
                    "Kali Linux & Network Security",
                    "OWASP Top 10 Web Security",
                    "Penetration Testing & Metasploit",
                    "SIEM & SOC Incident Response",
                    "Enterprise Security Audit Capstone",
                ],
                "readinessScore": readiness_score + 5,
                "summary": "Defensive security, ethical penetration testing, and enterprise vulnerability management.",
            }

        if has_dsa:
            return {
                "recommendedPath": "DSA & System Design Masterclass",
                "courses": [
                    "250+ LeetCode Patterns",
                    "Advanced Dynamic Programming & Graphs",
                    "Low-Level Design (SOLID & Design Patterns)",
                    "High-Level Distributed Systems Design",
                    "FAANG-style AI Mock Technical Interviews",
                ],
                "readinessScore": readiness_score + 15,
                "summary": "Competitive algorithmic problem solving and scalable distributed system design.",
            }

        return {
            "recommendedPath": "AI + Full Stack",
            "courses": [
                "Python",
                "Machine Learning",
                "Generative AI",
                "Java/Spring Boot",
                "React",
            ],
            "readinessScore": readiness_score,
            "summary": "Comprehensive foundation for software engineering and AI-driven applications.",
        }


def get_course_recommendations(profile: Dict[str, Any]) -> Dict[str, Any]:
    return CourseRecommendationService.get_recommendations(profile)
