# Was muss sein damit er still steht — Aktor-Einstellungen

**Bericht 3 der Domain-Insight-Reihe** | Quelle: MuJoCo Inverse Dynamics auf `home`-Keyframe

---

### 1. Kurze Antwort

„Still stehen" auf Aktor-Ebene bedeutet drei Dinge gleichzeitig:
**Soll-Pose halten** (qpos = home-Keyframe), **Schwerkraft kompensieren** (jeder Motor muss das über ihm hängende Gewicht gegen 9,81 m/s² halten), und **Mikro-Korrekturen** ausführen wenn der Schwerpunkt driftet.

Die unten gelisteten Drehmomente sind die **statischen Halte-Drehmomente** — die Mindestkraft, die jeder Motor aufbringen muss, nur um die Pose im Gleichgewicht zu halten.
Dynamische Balance-Korrekturen kommen **on-top**, je nach aktuellem Schwerpunkt-Offset.
Eine PPO-Policy, die still steht, schwankt um diese Werte herum — nicht um Null.

---

### 2. Soll-Tabelle — alle 19 Aktoren

Quelle Gelenkwinkel: `home`-Keyframe aus `scene.xml`, qpos[7..25].
Quelle Drehmomente: `mj_inverse(model, data)` mit qvel=0, qacc=0, gain=1.0 für alle Motor-Aktuatoren.

| Idx | Aktor-Name | Gruppe | Soll-Winkel (rad) | Soll-Winkel (deg) | Halte-Drehmoment (Nm) | ctrlrange | Auslastung (%) |
|-----|-----------|--------|--------------------|--------------------|-----------------------|-----------|----------------|
| 0 | left_hip_yaw | LOWER | 0.0000 | 0.0 | 0.000 | -200..+200 | 0.0 |
| 1 | left_hip_roll | LOWER | 0.0000 | 0.0 | -36.685 | -200..+200 | 18.3 |
| 2 | left_hip_pitch | LOWER | -0.4000 | -22.9 | 29.760 | -200..+200 | 14.9 |
| 3 | left_knee | LOWER | 0.8000 | 45.8 | -23.039 | -300..+300 | 7.7 |
| 4 | left_ankle | LOWER | -0.4000 | -22.9 | 33.039 | -40..+40 | 82.6 |
| 5 | right_hip_yaw | LOWER | 0.0000 | 0.0 | 0.000 | -200..+200 | 0.0 |
| 6 | right_hip_roll | LOWER | 0.0000 | 0.0 | 36.683 | -200..+200 | 18.3 |
| 7 | right_hip_pitch | LOWER | -0.4000 | -22.9 | 29.758 | -200..+200 | 14.9 |
| 8 | right_knee | LOWER | 0.8000 | 45.8 | -23.038 | -300..+300 | 7.7 |
| 9 | right_ankle | LOWER | -0.4000 | -22.9 | 33.037 | -40..+40 | 82.6 |
| 10 | torso | UPPER | 0.0000 | 0.0 | 0.000 | -200..+200 | 0.0 |
| 11 | left_shoulder_pitch | UPPER | 0.0000 | 0.0 | -1.097 | -40..+40 | 2.7 |
| 12 | left_shoulder_roll | UPPER | 0.0000 | 0.0 | 0.031 | -40..+40 | 0.1 |
| 13 | left_shoulder_yaw | UPPER | 0.0000 | 0.0 | 0.000 | -18..+18 | 0.0 |
| 14 | left_elbow | UPPER | 0.0000 | 0.0 | -1.044 | -18..+18 | 5.8 |
| 15 | right_shoulder_pitch | UPPER | 0.0000 | 0.0 | -1.097 | -40..+40 | 2.7 |
| 16 | right_shoulder_roll | UPPER | 0.0000 | 0.0 | -0.031 | -40..+40 | 0.1 |
| 17 | right_shoulder_yaw | UPPER | 0.0000 | 0.0 | 0.000 | -18..+18 | 0.0 |
| 18 | right_elbow | UPPER | 0.0000 | 0.0 | -1.044 | -18..+18 | 5.8 |

**Vorzeichen-Konvention:** positives Drehmoment = Gelenk in positiver Achsen-Richtung aufhalten. Für `left_hip_roll` (Achse X, negatives Drehmoment) bedeutet das: Motor zieht das Bein zur Mitte — das Bein fällt sonst nach links weg.

---

### 3. Anatomische Lesart

**Top-Auslastung:**

| Rang | Aktor | Drehmoment (Nm) | Auslastung | Erklärung |
|------|-------|-----------------|------------|-----------|
| 1 | left_ankle / right_ankle | ±33 Nm | **82.6 %** | Knöchel trägt Hauptlast gegen Vorwärts-Kippen — kleiner ctrlrange (40 Nm) |
| 2 | left_hip_roll / right_hip_roll | ±37 Nm | **18.3 %** | Hüfte hält Beine gegen seitliches Wegkippen |
| 3 | left_hip_pitch / right_hip_pitch | +30 Nm | **14.9 %** | Hüfte streckt Bein gegen leichte Vorlage |
| 4 | left_knee / right_knee | −23 Nm | 7.7 % | Knie hält gebeugten Winkel (0.8 rad = 46°) |

**Abgleich mit Erwartung:**

- **Knöchel-Pitch**: ≈ 33 Nm — Erwartung war „kleines Drehmoment". **Abweichung.** Grund: home-Keyframe hat Knöchel bei −0.4 rad (plantarflexion) und Knie bei +0.8 rad. Das bringt Körperschwerpunkt leicht nach vorn. Knöchel muss dagegen halten. Bei 82 % Auslastung ist das **der kritische Aktor** — wenig Reserve für dynamische Korrekturen.
- **Hüft-Roll**: ±37 Nm — Erwartung war ≈ 0 bei symmetrischer Pose. **Abweichung.** Grund: home-Pose hat breitbeinigen Stand (Hüft-Offset ±0.0875 m). Die Hüften stehen leicht außerhalb der Körpermasse-Projektion. Jede Seite muss das Beingewicht lateral halten.
- **Knie & Hüft-Pitch**: deutliche Werte ✅ — wie erwartet bei gebeugtem Kniestand.
- **Torso, Schultern, Ellbogen**: ≈ 0 bis max. 1.1 Nm ✅ — Arme hängen entspannt, Torso ist Rotationsneutral. Genau wie erwartet.

---

### 4. Verbindung zum aktuellen RL-Training

- Diese Tabelle zeigt den **passiven Gleichgewichtspunkt** der home-Pose. Eine PD-Policy bei perfekter Pose würde genau diese Drehmomente ausgeben.
- Die PPO-Policy aus Iter 5/6 schwingt um diese Werte herum. Je näher an diesen Werten, desto ruhiger der Stand.
- **Kritischer Befund:** Knöchel bei 82 % Auslastung. Wenn die Policy die Knöchel für Balance-Korrekturen nutzt (Ankle-Strategy), hat sie nur 7 Nm Reserve. Das erklärt das beobachtete Knöchel-Zappeln — die Policy drückt gegen die Grenze.
- **Möglicher Hebel (Action-Anchor-Term):** Reward-Komponente die die Policy-Action (ctrl) zu diesen statischen Soll-Drehmomenten zieht:

```python
# Action-Anchor: zieht Policy-Output zu statischen Halte-Drehmomenten
STATIC_TORQUE = np.array([0.0, -36.685, 29.760, -23.039, 33.039,
                           0.0,  36.683, 29.758, -23.038, 33.037,
                           0.0,  -1.097,  0.031,   0.000, -1.044,
                          -1.097, -0.031,  0.000,  -1.044])
ACTION_ANCHOR_WEIGHT = 0.001
anchor_cost = ACTION_ANCHOR_WEIGHT * np.sum((action - STATIC_TORQUE)**2)
```

Das ist **kein Pflicht-Vorschlag** — nur ein konkreter nächster Hebel, falls Iter 6 zu unruhig bleibt.

---

### 5. Reproduzierbar

```python
import mujoco, numpy as np

model = mujoco.MjModel.from_xml_path(
    "/mnt/data/Projects/Roboter/mujoco-test/mujoco_menagerie/unitree_h1/scene.xml"
)
data = mujoco.MjData(model)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
data.qvel[:] = 0.0
data.qacc[:] = 0.0
mujoco.mj_inverse(model, data)

for i in range(model.nu):
    name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jid   = model.actuator_trnid[i, 0]
    dadr  = model.jnt_dofadr[jid]
    qadr  = model.jnt_qposadr[jid]
    print(f"[{i:2d}] {name:30s} q={data.qpos[qadr]:7.4f} tau={data.qfrc_inverse[dadr]:9.3f} Nm")
```

Ausführen: `MUJOCO_GL=egl .venv-rl/bin/python <dieses_script.py>`
