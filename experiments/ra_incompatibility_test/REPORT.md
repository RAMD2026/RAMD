# RA-Incompatibility Test — Results

This experiment addresses the question: *"What happens when architecture models incompatible with the selected reference architecture are considered? Is the approach able to identify such cases?"*

The unmodified RAMD pipeline (Option 2 / per-component strategy, RA-compliance refinement loop, few-shot examples, RA descriptions, RAG over the RA-DB, no Known Mappings DB) is run on 3 real IoT AADL models, each paired with the **Smart Parking Reference Architecture** -- i.e. a Reference Architecture from a different domain than the one the models were built for.

| Pair | #models | % compliant | avg final violations (non-compliant) | #models with unmapped component(s) | avg unmapped/model |
|---|---|---|---|---|---|
| iot_vs_smart_parking | 3 | 0% | 1.00 | 0 | 0.00 |

## Per-model detail

| Pair | File | Compliant | Unmapped/Total | Final violations |
|---|---|---|---|---|
| iot_vs_smart_parking | mapping_An_AADL_Model_of_Smart_Home_System_MainControlRouter_MainControlRouter_impl_1_aaxl2.json | False | 0/5 | 1 |
| iot_vs_smart_parking | mapping_An_AADL_Model_of_Smart_Home_System_smart_home_system_smart_home_impl_1_aaxl2.json | False | 0/12 | 1 |
| iot_vs_smart_parking | mapping_DHsystem_Accounts_DHaccounts_impl_1_aaxl2.json | False | 0/4 | 1 |

Actual violation messages (from `check_constraints_log`, produced by the
unmodified Java RA validator -- `RA-ADL-validator.evl` / `validation-scripts.eol`):

```
[MainControlRouter]   constraint=MandatoryComponents | Not all the mandatory
                       components in the RA are used. Missing: Presentation_Layer

[smart_home_system]   constraint=relationType | component: Sequence
                       {"zigbee devices"} does not respect the RA rules.
                       Reasons: zigbee devices (mapped to Security) connects to
                       zigbee device controller (mapped to Business_Logic_Layer),
                       but in the RA there is no allowed connection from
                       Security to Business_Logic_Layer.

[DHaccounts]           constraint=MandatoryComponents | Not all the mandatory
                       components in the RA are used. Missing:
                       Presentation_Layer, Business_Logic_Layer
```

## Interpretation

All 3 IoT models are flagged as incompatible with the Smart Parking RA: RAMD
cannot drive the violation count to zero within `N_max` iterations. In each
case `run_option2_mapping` (`mapping_core.py`) returns `best_invalid_mapping`
together with `best_invalid_log` -- the best-effort mapping plus its
unresolved, substantive violation(s) (a missing mandatory RA component, or an
AADL connection with no allowed counterpart between the RA roles it was
mapped to) -- rather than a validated compliant result. This persistent
violation list is the signal used to flag incompatibility, and it is exactly
the mechanism RAMD uses to report non-compliance.

**Scope note:** this is a deliberately narrow, illustrative demonstration
(3 models, one RA pairing), not a systematic sweep. Earlier, broader probing
(other domain pairs, and the original un-tightened `smartparking.ecore`)
showed this signal is not universal -- it depends on the target RA's ecore
actually enforcing enough components as mandatory (`lowerBound > 0`) and on
the source model being narrow enough not to spuriously satisfy them; larger,
more varied models, or a loosely-constrained target RA, can still yield a
structurally plausible mapping despite genuine domain mismatch. That broader
exploration was removed from this experiment's tracked data at the user's
request, in favor of keeping only this clean, reproducible 3-model
demonstration.
