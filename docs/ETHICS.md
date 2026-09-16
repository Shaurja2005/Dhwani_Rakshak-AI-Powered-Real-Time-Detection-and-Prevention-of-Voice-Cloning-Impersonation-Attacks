# Ethics and consent

- Clone only consented speakers or public-domain corpora.
- Never clone a real executive, official, or non-consenting individual — including for testing.
- The generated corpus stays internal and is documented as defensive research.

## Consent register

| Name | Date | Scope of consent | Withdrawal contact |
|---|---|---|---|

## Enforcement (B3-T11)

The table above is for humans; `ml/data/consent_register.yaml` is its machine-readable mirror.
`ml/data/clone_job.py` refuses to clone any reference utterance whose corpus or speaker is not
listed there. Keep both in sync. Signed consent forms are stored outside git and referenced by
`form_ref`. Voices of real executives, officials or public figures must never be added (I9).
