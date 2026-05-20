# Domain-Insight-Agent — Wiederverwendbare Komponente in Olafs RL-Workflow

## Was ist das

Ein **Subagent mit klar definierter Rolle**: Biomechanik-Berater. Er bekommt:

- Frame-Streifen der aktuellen Policy
- Reward-Historie + Metriken (Drift, Gelenk-Speed, Episodenlänge)
- Robotermodell (XML mit Aktuator-Reihenfolge)
- Eine menschliche Heuristik vom Auftraggeber („was tut ein Mensch hier")

Und liefert:

- Diagnose des aktuellen Verhaltens (was sieht er in den Frames)
- Übersetzung der Heuristik in **konkrete Reward-Terme** mit Vorschlagswerten
- Begründung pro Term (Trace-Regel)
- Was **nicht** zu bestrafen ist (Überregulierungsfalle)
- Curriculum-Vorschlag

## Wann konsultieren

| Situation | Trigger |
|-----------|---------|
| Vor neuer Trainings-Iteration | Reward-Redesign steht an |
| Policy hängt im Schummeloptimum | „kann stehen, aber zappelt" / „kann laufen, aber driftet" |
| Neuer Skill soll gelernt werden | Aufstehen, Treppe, Greifen — bevor erster Code |
| Metriken plateau-en | 3+ Iterationen ohne Verbesserung der Ziel-Metrik |

**Nicht** konsultieren für:
- Hyperparameter-Sweeps (LR, batch_size)
- Bug-Hunt im Environment-Code
- PPO-Internas

## Wie konsultieren

Subagent mit Auftrag im Format:

```
Du bist Domain-Insight-Agent für <Roboter> RL.
Rolle: Biomechanik in Reward-Heuristiken übersetzen.
Kontext: <aktuelle Metriken + Reward-Historie>
Bilder: <Frame-Streifen-Pfade>
Modell: <XML-Pfad mit Aktuator-Reihenfolge>
Heuristik des Auftraggebers: "<O-Ton>"
Output: 3 Markdown-Dateien in `insights/`
```

Output-Konvention:
- `01_<skill>.md` Hauptbericht (800-1500 Wörter)
- `02_<naechster_skill>.md` Skizze für Folgeskill (kürzer)
- `README.md` Meta-Beschreibung (diese Datei)

Jeder Bericht endet mit konkreter Reward-Übersetzung (Python-Snippet + Gewichts-Vorschlag + Balance-Rechnung).

## Wo in den Karpathy-Workflow

```
1. Diagnose (Metriken + Frames)
       ↓
2. ➜ Domain-Insight-Agent ←────  HIER
       ↓
3. Reward-Redesign (mit konkreten Vorschlägen)
       ↓
4. TDD: Test rot → Implementieren → Test grün
       ↓
5. Train + verify
```

**Lückenfüller:** Zwischen „Roboter macht was Komisches" (Diagnose) und „wir bauen Penalty X ein" (Redesign) wurde bisher oft blind iteriert — Penalty raten, trainieren, gucken, wiederholen. Der Agent schließt diese Lücke mit einer **anatomischen Begründung pro Term**.

## Outputs aus Iteration 5/6

- [`01_stillstand.md`](01_stillstand.md) — H1 Stillstand-Insight, differenzierte Lower/Upper-Penalties + Foot-Lock
- [`02_aufstehen.md`](02_aufstehen.md) — H1 Aufsteh-Skizze, Multi-Contact-Bonus + Knee-Push

---

**Karpathy-Sanity:** Agent ist eine Rolle, kein Tool. Wiederverwendbar weil die Struktur (Diagnose → Heuristik → Reward-Übersetzung) Roboter-agnostisch ist.
