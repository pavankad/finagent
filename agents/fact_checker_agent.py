import os
import requests
from typing import Dict, List, Optional, Tuple, Union
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

# Google Fact Check Tools API endpoint
FACT_CHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

def fetch_fact_checks(claim: str) -> List[Dict]:
    """
    Query the Google Fact Check Tools API to find fact checks related to the claim.

    Args:
        claim: The claim to be fact-checked

    Returns:
        A list of fact check results or empty list if none found
    """
    if not API_KEY:
        raise ValueError("GOOGLE_FACT_CHECK_API_KEY environment variable not set")

    params = {
        "query": claim,
        "key": API_KEY
    }

    try:
        response = requests.get(FACT_CHECK_API, params=params)
        response.raise_for_status()

        # Parse the response
        result = response.json()
        claims = result.get("claims", [])
        return claims

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return []

def analyze_fact_checks(claims: List[Dict]) -> Tuple[str, float, List[Dict]]:
    """
    Analyze fact check results to determine verdict.

    Args:
        claims: List of fact check results from the API

    Returns:
        Tuple containing (verdict, confidence, evidence)
    """
    if not claims:
        return "Unverified", 0.0, []

    # Extract relevant information from each claim
    processed_claims = []
    ratings = []

    for claim in claims:
        # Extract the claim rating
        review_ratings = []
        for review in claim.get("claimReview", []):
            rating = review.get("textualRating", "")
            publisher = review.get("publisher", {}).get("name", "Unknown source")
            url = review.get("url", "")

            # Add to the list of ratings
            review_ratings.append({
                "rating": rating.lower(),
                "publisher": publisher,
                "url": url
            })

            # Determine if this rating suggests the claim is false or true
            rating_lower = rating.lower()
            if any(word in rating_lower for word in ["false", "fake", "incorrect", "misleading"]):
                ratings.append(-1)  # False claim
            elif any(word in rating_lower for word in ["true", "correct", "accurate"]):
                ratings.append(1)   # True claim
            else:
                ratings.append(0)   # Neutral or unclear

        processed_claims.append({
            "text": claim.get("text", ""),
            "reviews": review_ratings
        })

    # Calculate the overall verdict
    if not ratings:
        verdict = "Insufficient Information"
        confidence = 0.0
    else:
        avg_rating = sum(ratings) / len(ratings)
        if avg_rating <= -0.5:
            verdict = "False"
            confidence = min(1.0, abs(avg_rating))
        elif avg_rating >= 0.5:
            verdict = "True"
            confidence = min(1.0, abs(avg_rating))
        else:
            verdict = "Partially True"
            confidence = 0.5

    return verdict, confidence, processed_claims

def fact_check(claim: str) -> Dict:
    """
    Check if a claim has been fact-checked.

    Args:
        claim: The claim to be fact-checked

    Returns:
        A dictionary containing the verdict and supporting information
    """
    print(f"Checking claim: {claim}\n")

    # Fetch fact checks from the API
    fact_checks = fetch_fact_checks(claim)

    if not fact_checks:
        print("⚠️ No fact checks found for this claim")
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "evidence": [],
            "claim": claim
        }

    # Analyze the fact checks to determine a verdict
    verdict, confidence, evidence = analyze_fact_checks(fact_checks)

    print(f"✅ Verdict: {verdict} (confidence: {confidence:.2f})\n")
    print("📰 Top evidence:")

    for i, item in enumerate(evidence[:3]):
        print(f"{i+1}. {item['text']}")
        for review in item['reviews'][:2]:
            print(f"   - Rating: {review['rating']} (by {review['publisher']})")
            print(f"   - Source: {review['url']}\n")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "claim": claim
    }
if __name__ == "__main__":
    # Example usage
    claim = "The COVID-19 vaccine contains microchips that track your movements."
    result = fact_check(claim)

    print("\nFinal Result:")
    print(f"Claim: {result['claim']}")
    print(f"Verdict: {result['verdict']} (Confidence: {result['confidence']:.2f})")
    print("Evidence:")
    for item in result['evidence']:
        print(f"- {item['text']}")
        for review in item['reviews']:
            print(f"  - {review['rating']} by {review['publisher']} ({review['url']})")
    print("\nNote: This is a simulated fact-checking process using the Google Fact Check Tools API.")
