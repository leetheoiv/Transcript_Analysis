from data_models.HITL_datamodel import HITLDecision
def HITL(
    item_for_review,
    prompt: str = "Item for review. Please provide a decision.",
    actions: dict[str, str] | None = None,
    allow_notes: bool = True,
    default: str | None = None,
    ) -> HITLDecision:
        """
        Human-in-the-loop decision helper.

        Args:
            item_for_review: The item/content to display for review.
            prompt: Message shown before the review item.
            actions: Mapping of decision codes to human-readable labels.
                    Example: {"A": "Approve", "R": "Retry", "C": "Cancel"}
            allow_notes: If True, prompt reviewer for optional notes.
            default: Optional default decision if user presses Enter.

        Returns:
            HITLDecision
        """
        if actions is None:
            actions = {
                "A": "Approve",
                "R": "Retry",
                "C": "Cancel",
            }

        normalized_actions = {k.upper(): v for k, v in actions.items()}

        print(prompt)
        print("\n" + "-" * 70)

        action_display = ", ".join(
            [f"[{code}] - {label}" for code, label in normalized_actions.items()]
        )
        print(action_display)
        print("-" * 70)
        print("Item for Review:\n")
        print(item_for_review)
        print("-" * 70)

        while True:
            raw = input(
                f"Enter decision"
                + (f" [{default}]" if default else "")
                + ": "
            ).strip().upper()

            if not raw and default:
                raw = default.upper()

            if raw in normalized_actions:
                break

            print(
                f"Invalid input. Allowed values: {', '.join(normalized_actions.keys())}"
            )

        notes = ""
        if allow_notes:
            notes = input("Optional notes/comments: ").strip()

        return HITLDecision(
            decision=raw,
            decision_label=normalized_actions[raw],
            notes=notes,
            item_reviewed=str(item_for_review),
        )