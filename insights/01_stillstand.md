# Stillstand — Biomechanik-Insight für H1 RL (Iteration 6+)

**Beobachtung aus `final_25M.png`:** H1 steht aufrecht, breitbeinig, Arme leicht abgespreizt — aber über die Frame-Sequenz wandert der Torso sichtbar von links nach rechts. Knie und Hüfte wackeln mit. **Kein Stillstand, sondern „aktive Wackelbalance"**. Das ist exakt das Symptom hinter den Metriken (1,71 rad/s, 3,4 m Drift).

## 1. Was ist Stillstand — anatomisch

| Kriterium | Steht ohne zu fallen | Steht wirklich still |
|-----------|----------------------|----------------------|
| Füße | dürfen Schritte machen | bleiben **fest am Platz** (Kontakt + xy konstant) |
| Knöchel/Knie | beliebige Bewegung erlaubt | nur **kleinste Korrekturen** (< 0,2 rad/s) |
| Hüfte | dreht zur Balance | minimal, hauptsächlich Pitch/Roll |
| Torso/Schultern | egal | darf sich zur Schwerpunkt-Korrektur **mitbewegen** |
| Arme | egal | dürfen schwingen — sind aktive Balancer |

**Kernunterschied:** Beim echten Stillstand ist der **Kontaktpunkt mit der Welt (Fuß)** unverändert. Alles oberhalb ist Reaktion, nicht Aktion.

## 2. Menschliche Strategie — distal nach proximal

Die menschliche Balance arbeitet in einer klaren Reihenfolge:

| Reihenfolge | Strategie | Wann |
|-------------|-----------|------|
| 1. Knöchel-Strategie | leichte Fuß-/Wadenkorrektur | kleine Auslenkung (Center of Pressure innerhalb der Fußfläche) |
| 2. Hüft-Strategie | Hüfte kippt, Knie federt | mittlere Auslenkung |
| 3. Arm-Strategie | Arm-Schwung erzeugt Drehimpuls | große Auslenkung, kein Schritt nötig |
| 4. Schritt | Fuß setzt um | nur wenn alles andere versagt |

**Sensorik beim Menschen:**
- Druckverteilung Fußsohlen (= MuJoCo: `data.contact` + `cfrc_ext`)
- Vestibulärsystem (= IMU am Torso, schon im H1: `site name="imu"`)
- Propriozeption (= `qpos`, `qvel` jedes Gelenks)

**Regel:** Korrektur startet IMMER beim Fuß. Wenn ein Roboter zappelt, bevor der Fuß sich bewegt hat, hat er die Sensorik-Hierarchie umgedreht.

## 3. Olafs Heuristik konkret — „unten fest, oben balance"

Übersetzt in zwei Gelenk-Gruppen:

### Gruppe LOWER (= „unten fest")
Diese Gelenke tragen das Körpergewicht und definieren die Stand-Fläche. Müssen **ruhig** sein.

| Aktuator-Idx | Name | Begründung |
|--------------|------|------------|
| 0 | left_hip_yaw | rotiert Bein → verschiebt Fuß |
| 1 | left_hip_roll | seitliche Bein-Kippung |
| 2 | left_hip_pitch | vor/zurück Bein |
| 3 | left_knee | trägt Gewicht |
| 4 | left_ankle | Knöchel-Strategie — **kleine** Korrektur erlaubt |
| 5 | right_hip_yaw | spiegel |
| 6 | right_hip_roll | spiegel |
| 7 | right_hip_pitch | spiegel |
| 8 | right_knee | spiegel |
| 9 | right_ankle | spiegel |

→ **10 Gelenke. Strenger Velocity-Penalty.**

### Gruppe UPPER (= „oben darf balancieren")
Diese Gelenke korrigieren den Schwerpunkt durch Drehimpuls. Müssen **dürfen sich bewegen**.

| Aktuator-Idx | Name | Begründung |
|--------------|------|------------|
| 10 | torso | Twist um Vertikalachse — kleine Schwerpunktkorrektur |
| 11 | left_shoulder_pitch | Arm hoch/runter → Drehimpuls |
| 12 | left_shoulder_roll | Arm abspreizen → Trägheitsmoment vergrößern |
| 13 | left_shoulder_yaw | Arm rotieren |
| 14 | left_elbow | Arm beugen → Hebelarm anpassen |
| 15-18 | right_shoulder_*, right_elbow | spiegel |

→ **9 Gelenke. Milder Velocity-Penalty.**

**Warum die Trennung lebenswichtig ist:** Im aktuellen Iter-5-Reward bestraft `JOINT_VEL_COST_WEIGHT = 0.005 * sum(qvel²)` **alle 19 Gelenke gleich**. Damit der Roboter den Penalty unten reduziert, muss er auch oben starr werden — verliert dann aber den Balance-Trick und fällt. Resultat: Policy bleibt im Kompromiss bei 1,71 rad/s hängen. **Differenzieren statt skalieren.**

## 4. Reward-Übersetzung für `h1_stand_env.py` (Iteration 6)

### Differenzierter Joint-Velocity-Penalty

Ersetze in `h1_stand_env.py`:

```python
# Iter 5 (verwerfen):
# JOINT_VEL_COST_WEIGHT = 0.005
# joint_vel_cost = JOINT_VEL_COST_WEIGHT * np.sum(qvel[6:25]**2)

# Iter 6 (neu):
LOWER_JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])   # 10 Beine/Hüfte
UPPER_JOINT_IDX = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])  # 9 Torso/Arme

LOWER_JOINT_VEL_WEIGHT = 0.020   # 4x strenger als Iter-5 uniform
UPPER_JOINT_VEL_WEIGHT = 0.0005  # 10x milder

joint_qvel = qvel[6:25]
lower_vel_cost = LOWER_JOINT_VEL_WEIGHT * np.sum(joint_qvel[LOWER_JOINT_IDX]**2)
upper_vel_cost = UPPER_JOINT_VEL_WEIGHT * np.sum(joint_qvel[UPPER_JOINT_IDX]**2)
```

**Balance-Check bei aktueller Policy (1,71 rad/s über 19 Gelenke):**
- Unten zappelt — angenommen 1,5 rad/s × 10 Gelenke: `0.020 · 10 · 2.25 ≈ 0,45/Step` → klar dominant
- Oben darf — bei 2,0 rad/s × 9 Gelenke: `0.0005 · 9 · 4.0 ≈ 0,018/Step` → vernachlässigbar

**Balance-Check bei Zielzustand:**
- Unten 0,2 rad/s × 10: `0.020 · 10 · 0.04 ≈ 0,008` → erlaubt
- Oben 1,0 rad/s × 9: `0.0005 · 9 · 1.0 ≈ 0,005` → erlaubt

Klare Trennung: Bein-Zappeln teuer, Arm-Schwung billig.

### Foot-Position-Lock-Term (NEU — der Kern von Olafs Heuristik)

Speichere im `reset()` die Reset-xy-Position beider Füße. Bestrafe Abweichung:

```python
# in reset():
self._foot_xy_init = np.array([
    self.data.body("left_ankle_link").xpos[:2].copy(),
    self.data.body("right_ankle_link").xpos[:2].copy(),
])

# in step() reward:
FOOT_POS_LOCK_WEIGHT = 2.0
left_foot_drift  = np.linalg.norm(self.data.body("left_ankle_link").xpos[:2]  - self._foot_xy_init[0])
right_foot_drift = np.linalg.norm(self.data.body("right_ankle_link").xpos[:2] - self._foot_xy_init[1])
foot_lock_cost = FOOT_POS_LOCK_WEIGHT * (left_foot_drift + right_foot_drift)
```

**Begründung Gewicht 2.0:** Linear. Bei 5 cm Fuß-Drift pro Fuß → −0,2/Step (erlaubt für Korrektur). Bei 30 cm → −1,2/Step (klar bestraft). Bei 1 m (= aktueller Wander-Modus) → −4,0/Step — dominant, killt das Wandern direkt.

### Foot-Contact-Penalty (NEU — Füße sollen Boden behalten)

Füße abheben = kein Stillstand. Über MuJoCo-Kontakte:

```python
def _both_feet_on_ground(self):
    left_id  = self.model.body("left_ankle_link").id
    right_id = self.model.body("right_ankle_link").id
    left_contact = right_contact = False
    for i in range(self.data.ncon):
        c = self.data.contact[i]
        b1 = self.model.geom_bodyid[c.geom1]
        b2 = self.model.geom_bodyid[c.geom2]
        if left_id  in (b1, b2): left_contact  = True
        if right_id in (b1, b2): right_contact = True
    return left_contact, right_contact

FOOT_AIR_PENALTY = 0.5
left_c, right_c = self._both_feet_on_ground()
foot_air_cost = FOOT_AIR_PENALTY * ((not left_c) + (not right_c))
```

**Begründung 0.5:** Ein Fuß einen Step in der Luft → −0,5. Beide → −1,0. Spürbar, aber nicht so groß dass jeder Mikro-Sprung sofort die Policy crasht — RL braucht weiches Signal.

### Bestehende Terme behalten

| Term | Status | Begründung |
|------|--------|------------|
| Pose-Referenz (POSE_WEIGHT=2.0, SHARPNESS=6.0) | ✅ unverändert | Funktioniert, ist der Anker zur home-Pose |
| ALIVE_BONUS=0.5 | ✅ unverändert | Stabil |
| UPRIGHT_WEIGHT=0.3 | ✅ unverändert | Stütz-Term |
| ORIENTATION_WEIGHT=0.3 | ✅ unverändert | Stütz-Term |
| CONTROL_COST=0.001 | ✅ unverändert | Mild gegen Brute-Force-Drehmoment |
| XY_DRIFT_WEIGHT=0.5 | ⚠️ redundant zu foot_lock — **entfernen** | foot_lock ist präziser (misst Fuß, nicht Pelvis) |
| ROOT_LIN_VEL_WEIGHT=0.3 | ✅ behalten | Geschwindigkeitsbasiert, ergänzt Position |
| JOINT_VEL_COST_WEIGHT=0.005 (uniform) | ❌ ersetzen | durch differenzierte Lower/Upper |

### Reward-Summe — globales Optimum

Ruhige Statue (Füße fest, Beine still, Arme dürfen leicht schwingen):
- pose 2.0 + alive 0.5 + upright 0.3 + orientation 0.3 = **3.1**
- Penalties: lower_vel ≈ 0.008, upper_vel ≈ 0.005, foot_lock ≈ 0, foot_air ≈ 0, root_lin ≈ 0.03, control ≈ 0
- **Netto ≈ 3.05/Step** ← globales Maximum

Aktueller Zustand (wandert):
- pose ≈ 1.9 + 0.5 + 0.3 + 0.3 = 3.0
- lower_vel ≈ 0.45, upper_vel ≈ 0.02, foot_lock ≈ 4.0, root_lin ≈ 0.1
- **Netto ≈ −1.6/Step** ← klar negativ, Policy-Gradient zeigt scharf weg

## 5. Was NICHT bestrafen — explizite Liste

Damit das Training nicht überreguliert wird und der Roboter weiter balancieren darf:

| Erlaubt | Warum |
|---------|-------|
| Arm-Schwung (Schulter Pitch/Roll/Yaw, Ellbogen) | aktiver Drehimpuls-Balancer |
| Leichte Torso-Twist | Schwerpunkt-Korrektur über Rumpf |
| Knöchel-Mikrokorrektur (< 0,3 rad/s) | echte Ankle-Strategy |
| Leichte Knie-Federung | dämpft Stöße |
| Pelvis-Pitch/Roll bis ~5° | Hüft-Strategy |
| Torso-Höhe-Schwankung ≤ 1 cm | natürliches Schwingen |

→ Keinen Penalty auf Arm-Position, keinen auf Torso-Höhen-Varianz, keinen auf Hüft-Pitch.

## 6. Curriculum-Vorschlag

**Lineares Anziehen der neuen Terme über 5M Steps:**

```python
progress = min(1.0, self._total_steps / 5_000_000)
foot_lock_w = 2.0 * progress
lower_vel_w = 0.005 + (0.020 - 0.005) * progress  # startet wie Iter 5
```

**Begründung:** Bei vollem Gewicht ab Step 0 wäre der Initial-Reward stark negativ (die Iter-5-Policy wandert ja). PPO würde versuchen, die Penalties durch Hinfallen zu vermeiden („Game Over = keine weiteren Penalties"). Linear hochziehen vermeidet diesen Death-Wish-Bug.

**Alternative (einfacher):** Resume von Iter-5-Checkpoint, neue Penalties **sofort voll**, aber `ent_coef` von 0.0 auf 0.01 für 1M Steps anheben um die Policy aus dem Wackel-Optimum zu schubsen.

---

**Karpathy-Sanity:** Trace-Regel ✅ — jeder neue Term lässt sich direkt auf Olafs Heuristik („unten fest, oben balance") oder eine konkrete Beobachtung im Frame-Streifen zurückführen. Senior-Check ✅ — kein neuer Mechanismus den ein PPO-erfahrener Engineer nicht sofort umsetzt. Trennung in 2 Gruppen + Foot-Lock sind die zwei chirurgischen Eingriffe, nicht 7.
