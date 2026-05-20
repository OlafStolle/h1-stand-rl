# Aufstehen — Biomechanik-Insight für H1 RL (Phase 2)

**Beobachtung:** Phase 2 startet aus Downed-Pose (NOTES.md Iter Phase 2). Aktueller Reward ist Pose-Tracking + `height_gain`. Funktioniert prinzipiell, aber ohne Strategie-Anker — Policy findet irgendeinen Weg hoch, oft nicht den biomechanisch sinnvollen. Hier eine Strategie-Skizze.

## 1. Menschliche Aufsteh-Strategien

| Start-Lage | Wege | Wann gewählt |
|------------|------|--------------|
| Bauchlage (prone) | → Vierfüßler → kniender Ausfallschritt → Stand | Kraftvoll, Standard für trainierte Menschen |
| Bauchlage | → Liegestütz hoch → Hocke → Stand | Athletisch, viel Armkraft |
| Rückenlage (supine) | → Roll auf Seite → Bauchlage → s.o. | Wenn Bauchmuskeln schwach |
| Rückenlage | → Sit-up → Hocke → Stand | Wenn Rumpf stark |
| Seitlage | → Stütz auf Ellbogen → Vierfüßler → Stand | Übergangsweg |

**Gemeinsame Phasen** (= curriculum-Stufen):

1. **Orientierung** — Kopf hoch, Vestibulärsystem checkt Schwerkraft-Richtung
2. **Boden-Kontakt aufbauen** — Hände/Knie/Füße als breite Stützfläche
3. **Schwerpunkt anheben** — über Stütz-Fläche
4. **Schwerpunkt nach vorn über Füße verlagern** — Hände lösen
5. **Aufrichten** — Knie strecken, Hüfte strecken, Torso senkrecht

## 2. Anwender-Heuristik übertragen — „wo gehören die Gliedmaßen hin"

| Phase | Funktion der Beine | Funktion der Arme |
|-------|---------------------|---------------------|
| 1 (liegend) | passiv | Kopf abstützen, Boden tasten |
| 2 (Vierfüßler) | Knie als Stütze | Hände als Stütze, breit |
| 3 (Knien) | beide Knie tragen | Arme balancieren |
| 4 (Halbstand) | ein Fuß vor, Knie streckt | Arme schwingen vor, Drehimpuls |
| 5 (Stand) | beide Füße tragen | Arme entspannen, Phase-1-Policy übernimmt |

**Roboter-Vorteile** gegenüber Mensch:
- Symmetrischer Push aus beiden Armen gleichzeitig möglich
- Knie können bis 2,05 rad gebeugt werden (sehr tief in die Hocke)
- Keine Schmerzen → aggressivere Bewegungen erlaubt

**Roboter-Nachteile:**
- Schultern haben nur 40 Nm ctrlrange — KEIN klassischer Liegestütz-Push
- Knie haben 300 Nm — der **starke** Aktuator
- → Beine-getriebene Strategie ist mechanisch passender als Arm-Push

## 3. Reward-Skizze für `h1_standup_env.py` v2

Phase-2-Reward ist aktuell zu „blind": belohnt nur Höhe und Pose-Nähe, nicht den **Pfad**. Vorschlag: Phasen-aware Stütz-Reward.

### Term 1 — Multi-Contact-Bonus (NEU)

Vierfüßler-Stütze (Hände + Knie + Füße = 4-6 Kontakte) ist eine sinnvolle Zwischenstufe. Belohne **viele Kontaktpunkte mit dem Boden in der Aufricht-Phase**:

```python
def _ground_contacts(self):
    """Zählt Bodies, die den Boden berühren — Hände, Knie, Füße, Pelvis."""
    candidate_bodies = ["left_ankle_link", "right_ankle_link",
                        "left_knee_link", "right_knee_link",
                        "left_elbow_link", "right_elbow_link",
                        "pelvis"]
    ids = {b: self.model.body(b).id for b in candidate_bodies}
    touched = set()
    floor_id = self.model.geom("floor").id if any(
        self.model.geom(i).name == "floor" for i in range(self.model.ngeom)
    ) else 0
    for i in range(self.data.ncon):
        c = self.data.contact[i]
        for b in candidate_bodies:
            if ids[b] in (self.model.geom_bodyid[c.geom1],
                          self.model.geom_bodyid[c.geom2]):
                touched.add(b)
    return touched

# Reward:
MULTI_CONTACT_WEIGHT = 0.1
# nur belohnen wenn Torso noch tief — also Stütz-Phase
contacts = self._ground_contacts()
if qpos[2] < 0.6:
    multi_contact = MULTI_CONTACT_WEIGHT * len(contacts)
else:
    multi_contact = 0  # in Stehnähe: keine zusätzlichen Kontakte mehr belohnen
```

**Begründung:** Belohnt Vierfüßler-Übergangslage in der frühen Phase. Schaltet ab, sobald Roboter über Halbstand hinaus ist — sonst lernt er kriechen statt aufzustehen.

### Term 2 — Asymmetrie-Penalty bei tiefer Lage entfernen

Aktuell zieht `pose_error` schon im Liegen scharf an die symmetrische home-Pose. Aber das Vierfüßler ist asymmetrisch (Hüfte gebeugt). Lösung: Pose-Sharpness phase-abhängig:

```python
# Iter Phase 2 v2:
if qpos[2] < 0.4:
    pose_sharpness = 1.0   # sehr mild, Pose-Anker fast aus
elif qpos[2] < 0.7:
    pose_sharpness = 2.0   # mittel
else:
    pose_sharpness = 6.0   # voll wie Phase 1

pose_reward = POSE_WEIGHT * np.exp(-pose_sharpness * pose_error)
```

**Begründung:** Liegt der Torso unten, ist die home-Pose physikalisch unerreichbar in einem Step. Scharfer Penalty erzeugt nahezu 0-Gradient → kein Lernsignal. Milde Sharpness lässt die Policy explorieren.

### Term 3 — Knie-Push-Bonus

H1s stärkster Motor ist das Knie (300 Nm). Klassische menschliche Aufricht-Strategie nutzt Knie-Streckung. Belohne **Knie-Streckung wenn beide Füße Boden haben und Torso über 0,5 m**:

```python
KNEE_EXTENSION_WEIGHT = 0.3
left_knee_angle  = qpos[7 + 3]  # left_knee = 4. Gelenk → qpos-Idx 10
right_knee_angle = qpos[7 + 8]  # right_knee = 9. Gelenk → qpos-Idx 15
both_feet = all(self._both_feet_on_ground())
if both_feet and qpos[2] > 0.5:
    # 0 rad = gestreckt, 2.05 rad = max gebeugt → belohne kleine Werte
    knee_extension = KNEE_EXTENSION_WEIGHT * (
        (2.05 - left_knee_angle) + (2.05 - right_knee_angle)
    ) / 4.10
else:
    knee_extension = 0
```

### Curriculum-Stufen

| Stufe | Steps | Reset-Höhe | Ziel |
|-------|-------|------------|------|
| 2a | 0-5M | Vierfüßler (Hocke, Hände frei) | Stand erreichen |
| 2b | 5-10M | Hocke (squat) | Stand |
| 2c | 10-20M | Bauchlage | Vierfüßler → Stand |
| 2d | 20-30M | Rückenlage | volle Sequenz |

Statt Phase 2 sofort aus voller Bauchlage zu trainieren: **Reset-Höhe schrittweise senken**. PPO konvergiert in jeder Stufe zu sub-Skill, der nächste Stufe einfacher macht.

## 4. Was später noch dazukommen sollte

| Term | Wann |
|------|------|
| Hand-Boden-Kontakt-Termination (verhindert Dauer-Kriechen) | wenn Roboter „mogelt" |
| Symmetrie-Bonus links/rechts | wenn Policy einseitig hochkommt |
| Drehimpuls-Penalty um Hochachse | wenn Roboter beim Aufstehen rotiert |

---

**Karpathy-Sanity:** Skizze, keine fertige Reward-Formel. Phase 1 muss erst stillstehen (Iter 6), dann Phase 2 angehen. Trace-Regel: jeder Term auf eine konkrete Aufsteh-Phase oder H1-Hardware-Eigenschaft (Knie-Drehmoment) bezogen.
