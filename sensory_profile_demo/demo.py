"""
AI Sensory Profile Designer — Demo
Generates a personalized sensory accommodation plan using the Claude API.
"""

import anthropic
import json


# ── Sample intake (simulating a filled-out form) ───────────────────────────────
SAMPLE_PROFILE = {
    "person": {
        "name": "Alex",
        "age": 9,
        "diagnosis": "Autism Spectrum Disorder (Level 1)",
        "primary_environment": "home + school"
    },
    "sensory_responses": {
        "auditory": {
            "sensitivity": "high",
            "triggers": [
                "loud unexpected noises (vacuum, fire alarm)",
                "multiple people talking at once",
                "high-pitched sounds"
            ],
            "notes": "Covers ears frequently in the cafeteria. Wears headphones at home."
        },
        "visual": {
            "sensitivity": "moderate",
            "triggers": [
                "flickering fluorescent lights",
                "bright overhead lighting",
                "crowded visual environments"
            ],
            "notes": "Squints under classroom lights. Prefers dim rooms for homework."
        },
        "tactile": {
            "sensitivity": "high",
            "triggers": [
                "tags in clothing",
                "certain fabric textures (scratchy wool, stiff denim)",
                "unexpected touch from behind"
            ],
            "notes": "Only wears seamless socks. Meltdowns when clothing feels wrong."
        },
        "proprioceptive": {
            "seeking": True,
            "behaviors": [
                "constantly moving or fidgeting",
                "crashes into furniture",
                "seeks heavy work (carrying backpack, pushing)"
            ],
            "notes": "Calms down after physical activity. Likes weighted blanket at night."
        },
        "vestibular": {
            "sensitivity": "low",
            "behaviors": [
                "swings for long periods",
                "spins without dizziness",
                "jumps frequently"
            ]
        }
    },
    "environments": {
        "home": {
            "rooms": ["bedroom", "kitchen", "living room"],
            "known_issues": [
                "kitchen has fluorescent strip lights",
                "open-plan living area is loud during family time",
                "bedroom has no dedicated calm-down space"
            ]
        },
        "school": {
            "grade": "3rd grade",
            "known_issues": [
                "cafeteria noise is overwhelming",
                "classroom has 28 students",
                "PE class changes are chaotic"
            ]
        }
    },
    "current_supports": [
        "noise-cancelling headphones (own)",
        "weighted blanket at home"
    ]
}


SYSTEM_PROMPT = """You are an expert occupational therapy assistant trained in sensory processing and
Dunn's Sensory Processing Framework. You help families and educators create practical, evidence-based
sensory accommodation plans.

IMPORTANT: You provide planning and accommodation suggestions only — NOT clinical diagnoses.
Always recommend follow-up with a licensed occupational therapist.

When generating a Sensory Accommodation Plan:
1. Acknowledge the person's sensory profile clearly
2. Organize recommendations by environment (home, school)
3. For each recommendation be SPECIFIC (exact products, measurements, placement)
4. Include a "Quick Wins" section — changes that cost under $20 and can be done today
5. Include a "Sensory Diet" — a daily schedule of sensory activities
6. End with a "For the OT" section — a concise summary for professional handoff

Keep language accessible for parents and teachers, not clinical."""


def generate_accommodation_plan(profile: dict) -> str:
    client = anthropic.Anthropic()

    print("Generating sensory accommodation plan...\n")

    full_response = ""

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Please generate a full Sensory Accommodation Plan for the following profile:

{json.dumps(profile, indent=2)}

Format the output clearly with headers and bullet points. Be specific and actionable."""
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    return full_response


def main():
    print("=" * 60)
    print("   AI SENSORY PROFILE DESIGNER — Demo")
    print("=" * 60)
    print(f"\nClient: {SAMPLE_PROFILE['person']['name']}, age {SAMPLE_PROFILE['person']['age']}")
    print(f"Diagnosis: {SAMPLE_PROFILE['person']['diagnosis']}")
    print(f"Environments: {SAMPLE_PROFILE['person']['primary_environment']}")
    print("\n" + "-" * 60 + "\n")

    plan = generate_accommodation_plan(SAMPLE_PROFILE)

    print("\n\n" + "=" * 60)
    print("   DISCLAIMER")
    print("=" * 60)
    print("This plan was generated by an AI assistant for planning purposes.")
    print("It is NOT a clinical assessment. Please consult a licensed")
    print("Occupational Therapist before implementing significant changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
