# H1 Stand — RL Design Notes

Ziel: Unitree H1 Humanoid lernt selbstständig zu stehen (Balance halten, nicht
umfallen) via PPO in MuJoCo.

## Modell

- `scene.xml` inkludiert `h1.xml` aus dem `mujoco_menagerie`.
- 19 Torque-Motoren (Actuators), `nq=26`, `nv=25`.
- `freejoint` der Pelvis → Root: 3 Position + 4 Quaternion.
- Keyframe `home`: aufrechter Stand, Torso-Höhe `qpos[2] = 0.98`.

### qpos-Layout (26)
| Index | Inhalt |
|-------|--------|
| 0:3   | Root-Position (x, y, z) |
| 3:7   | Root-Quaternion (w, x, y, z) |
| 7:26  | 19 Gelenkwinkel |

### qvel-Layout (25)
| Index | Inhalt |
|-------|--------|
| 0:3   | Root-Linear-Velocity |
| 3:6   | Root-Angular-Velocity |
| 6:25  | 19 Gelenkgeschwindigkeiten |

## Environment (`h1_stand_env.py`)

- `H1StandEnv(gymnasium.Env)`, Pfad zu `scene.xml` robust per `os.path` relativ
  zur Datei aufgelöst.
- `frame_skip = 5` → Control-Timestep ≈ 5 × Physics-dt (~0.01 s).

### Observation — Box float32, Dim 49
| Block | Quelle | Größe |
|-------|--------|-------|
| Torso-Höhe | `qpos[2]` | 1 |
| Root-Quaternion | `qpos[3:7]` | 4 |
| Gelenkwinkel | `qpos[7:]` | 19 |
| Geschwindigkeiten | `qvel[:]` | 25 |

Root-x/y bewusst weggelassen — die absolute Translation ist für die
Steh-Aufgabe irrelevant (Translations-Invarianz).

### Action — Box [-1, 1], Dim 19

Normalisierte Aktion wird linear auf den `ctrlrange` jedes Motors abgebildet:

```
ctrl = ctrl_low + 0.5 * (action + 1) * (ctrl_high - ctrl_low)
```

ctrlrange-Bereiche aus dem Modell: Hüfte/Torso ±200 Nm, Knie ±300 Nm,
Ankle/Schulter ±40 Nm, Schulter-Yaw/Elbow ±18 Nm.

### Reward pro Step — Iteration 1 (VERWORFEN)

```
reward = ALIVE_BONUS
       + UPRIGHT_WEIGHT     * exp(-HEIGHT_SHARPNESS * (z - TARGET_HEIGHT)^2)
       + ORIENTATION_WEIGHT * max(0, uprightness)
       - CONTROL_COST_WEIGHT * sum(action^2)
```

| Konstante | Wert |
|-----------|------|
| `ALIVE_BONUS` | 1.0 |
| `UPRIGHT_WEIGHT` | 1.0 |
| `HEIGHT_SHARPNESS` | 12.0 |
| `ORIENTATION_WEIGHT` | 1.0 |
| `CONTROL_COST_WEIGHT` | 0.001 |

**Ergebnis nach 3M Steps: gescheitert.** H1 steht nicht — 0 % volle
Episoden, mittlere Episodenlänge nur 159 Schritte.

**Warum Iteration 1 scheiterte:** Der Reward bestrafte Bewegung nur über
einen winzigen `control_cost`. Er belohnte ausschließlich *Torso-Höhe* und
*Torso-Aufrichtung* — beides sagt nichts über die *Gelenkstellung* aus. Die
Policy fand ein lokales Optimum: mit übertriebenen, zappelnden Gelenk- und
Armbewegungen den Torso kurzfristig hoch und aufrecht halten. Dieses Zappeln
ist instabil und führt nach ~159 Schritten zum Sturz. Es gab kein
Reward-Signal, das *ruhiges Halten der Stehpose* belohnt.

### Reward pro Step — Iteration 2: Pose-Referenz-Reward (AKTIV)

Etablierter Ansatz für Humanoid-Standing: **Pose-Tracking**. Eine
Referenzpose (die `home`-Stehpose) wird vorgegeben; der dominante
Reward-Term belohnt die Nähe der 19 Gelenkwinkel zu dieser Pose. Damit ist
*ruhiges Stehen in der Zielpose* — nicht Zappeln — das globale Optimum.

```
pose_error = mean((joint_qpos - home_joint_qpos)^2)   # über 19 Gelenke

reward =  POSE_WEIGHT        * exp(-POSE_SHARPNESS   * pose_error)        # DOMINANT
        + ALIVE_BONUS
        + UPRIGHT_WEIGHT     * exp(-HEIGHT_SHARPNESS * (z - TARGET_HEIGHT)^2)
        + ORIENTATION_WEIGHT * max(0, uprightness)
        - CONTROL_COST_WEIGHT   * sum(action^2)
        - JOINT_VEL_COST_WEIGHT * sum(joint_qvel^2)
```

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `POSE_WEIGHT` | 2.0 | **Dominanter Term** — Nähe zur `home`-Gelenkpose |
| `POSE_SHARPNESS` | 6.0 | Schärfe der Gauss-Glocke um die Referenzpose |
| `ALIVE_BONUS` | 0.5 | Bonus pro überlebtem Step |
| `UPRIGHT_WEIGHT` | 0.3 | Stütz-Term — Torso-Höhe nahe Ziel |
| `HEIGHT_SHARPNESS` | 12.0 | Schärfe der Höhen-Gauss-Glocke |
| `TARGET_HEIGHT` | 0.98 | Ziel-Torso-Höhe (= Keyframe `home`) |
| `ORIENTATION_WEIGHT` | 0.3 | Stütz-Term — Torso vertikal |
| `CONTROL_COST_WEIGHT` | 0.001 | Strafe für Stellgrößen-Aufwand |
| `JOINT_VEL_COST_WEIGHT` | 0.0005 | Milde Strafe gegen Zappeln (Gelenk-Geschw.) |

- **Referenzpose**: `model.key("home").qpos[7:26]` — die 19 Gelenkwinkel der
  `home`-Keyframe (qpos[0:7] = Root-Position + Quaternion, danach 19 Gelenke).
  Beim Env-Init einmalig als `self._home_joint_qpos` gespeichert.
- **pose_reward**: `2.0 * exp(-6 * mean((q - q_home)^2))` — maximal (= 2.0)
  bei exakter Stehpose, fällt mit dem mittleren quadratischen Gelenkfehler ab.
  Dominiert die Reward-Summe.
- **upright_reward / orientation_reward**: jetzt nur noch Stütz-Terme
  (Gewicht 0.3 statt 1.0) — verhindern Schummel-Posen, die zwar die
  Gelenkpose treffen, aber den Torso kippen.
- **control_cost**: quadratische Strafe auf der normalisierten Aktion.
- **joint_vel_cost**: quadratische Strafe auf den 19 Gelenkgeschwindigkeiten
  (`qvel[6:25]`) — drückt aktiv gegen schnelles Zappeln.

**Reward-Balance (Begründung der Gewichtswahl):** Ruhiges Stehen exakt in
der `home`-Pose ergibt ≈ `2.0 + 0.5 + 0.3 + 0.3 = 3.1` pro Step bei
nahezu null Kosten — das ist klar das globale Maximum. Jede Abweichung der
Gelenke senkt sofort den dominanten `pose`-Term; Zappeln senkt ihn zusätzlich
über `joint_vel_cost`. Das Iteration-1-Schummel-Optimum (hoher Torso durch
wilde Bewegung) liefert nun einen kleinen `pose`-Term und wird bestraft —
es ist kein Optimum mehr.

### Termination
| Bedingung | Typ | Wert |
|-----------|-----|------|
| Torso-Höhe < `FALL_HEIGHT` | `terminated` | 0.5 (umgefallen) |
| Step-Zähler ≥ `MAX_EPISODE_STEPS` | `truncated` | 1000 |

### reset()
- `mujoco.mj_resetDataKeyframe` auf Keyframe `home`.
- Gauss-Noise mit Std `RESET_NOISE = 0.01` auf `qpos` und `qvel`.
- `mujoco.mj_forward` zum Konsistent-Setzen abgeleiteter Größen.

### render()
- Offscreen-RGB-Frame (640×480) via `mujoco.Renderer`.

## Training (`train.py`)

- Algorithmus: **PPO** (`MlpPolicy`) aus stable-baselines3.
- `SubprocVecEnv` mit `n_envs` parallelen Envs, gewrappt in `VecMonitor`.

| Hyperparameter | Wert |
|----------------|------|
| `n_steps` | 2048 |
| `batch_size` | 512 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `ent_coef` | 0.0 |
| `learning_rate` | 3e-4 |
| `net_arch` | [256, 256] |

- CLI-Defaults: `--timesteps 3_000_000`, `--n-envs 12`, `--record-freq 200_000`.
- TensorBoard-Logs nach `tb/`.
- Callbacks:
  - `RecordCallback` — Greedy-Rollout-Video alle `record-freq` Steps
    (+ Baseline-Video bei Step 0).
  - `CheckpointCallback` — Checkpoint alle 500k Steps nach `checkpoints/`
    (`save_freq = 500_000 // n_envs`, da pro `_on_step` n_envs Steps zählen).
- Abschluss: `model.save("checkpoints/h1_stand_final")`.
- `multiprocessing`-safe via `if __name__ == "__main__"`.

## Video-Aufnahme (`record_callback.py`)

- Custom `BaseCallback`, rendert ein vollständiges deterministisches Rollout
  (bis terminated/truncated, max. 1000 Steps) und speichert es als
  `videos/versuch_{steps:07d}.mp4` (imageio, 30 fps).
- Offscreen-GL: `os.environ["MUJOCO_GL"]` wird auf `egl` gesetzt; schlägt die
  Renderer-Erzeugung fehl, Fallback auf `osmesa`. Funktioniert ohne Display.

## Live-Viewer (`watch.py`)

- Lädt `checkpoints/h1_stand_final.zip` (oder neuesten Checkpoint nach mtime).
- `mujoco.viewer.launch_passive`, fährt die Policy deterministisch und synct
  den Viewer in Echtzeit (Sleep auf Control-dt = `timestep * FRAME_SKIP`).

## Verifikation

- `check_env(H1StandEnv())` — ohne Fehler.
- Smoke-Test `train.py --timesteps 5000 --n-envs 2 --record-freq 2500` —
  vollständig durchgelaufen, MP4s erzeugt (inkl. Baseline bei Step 0).

---

# Phase 2 — Aufstehen (`h1_standup_env.py`)

Phase 1 (Stehen halten) wurde abgeschlossen: H1 steht zu 83 % nach 25M
Schritten. Finaler Checkpoint: `checkpoints/h1_stand_final.zip`. Phase 2
trainiert eine separate Policy darauf, **aus einer am Boden liegenden
Startpose selbständig in den Stand zu kommen**.

`h1_stand_env.py` bleibt unangetastet — Phase 2 ist eine eigene
`H1StandupEnv`-Klasse in `h1_standup_env.py`.

## Designentscheidungen

### Reset = "downed" Pose statt aufrecht

Statt im Stand zu starten und Balance zu halten, startet die Episode mit dem
Roboter zusammengesackt am Boden. Implementierung robust via natürlichem
Physikfall:

- Einmalig im `__init__`: `home`-Keyframe laden, `ctrl = 0`, dann
  `DOWNED_SETTLE_STEPS = 200` Simulationsschritte (= 200 × FRAME_SKIP = 1000
  Physik-Steps) mit Null-Drehmoment laufen lassen. Der Roboter kippt
  natürlich um und kommt zur Ruhe.
- Den resultierenden `qpos/qvel` als `self._downed_qpos`/`self._downed_qvel`
  (Deep-Copy) speichern.
- Bei jedem `reset()`: diesen Downed-State setzen + Gaussian-Noise mit
  `RESET_NOISE = 0.02` (etwas größer als Phase 1, damit variierte
  Start-Liegepositionen entstehen).

**Warum so?** Eine "downed-Pose" hartzukodieren wäre fragil (welche Gelenke
sind verbogen? wie liegen die Beine?). Die Physik liefert eine realistische,
reproduzierbare Liegepose; vom selben Reset-Skript erzeugt = identisch über
alle Workers.

### Keine Termination bei niedriger Torso-Höhe

Phase 1 brach bei Torso-z < 0,5 ab (umgefallen). Für Aufstehen wäre das
fatal: der Roboter startet ja niedrig. Daher:

- `terminated = False` immer.
- `truncated` nach `MAX_EPISODE_STEPS = 1500` (länger als Phase 1, weil
  Aufstehen Zeit braucht).

### Reward — selbe Struktur wie Phase 1, plus Höhe-Gain

Die Pose-Referenz (`home`-Keyframe-Gelenkwinkel) wird übernommen — sie
bleibt das Endziel. Damit der Lernsignal-Gradient aus der weit entfernten
Liegepose überhaupt brauchbar ist, werden Schärfe und Gewichte angepasst:

```
pose_error  = mean((joint_qpos - home_joint_qpos)^2)
height_gain = max(0, z - z_prev)        # nur positiver Gewinn

reward =  POSE_WEIGHT        * exp(-POSE_SHARPNESS * pose_error)          # DOMINANT
        + ALIVE_BONUS
        + UPRIGHT_WEIGHT     * exp(-HEIGHT_SHARPNESS * (z - TARGET_HEIGHT)^2)
        + ORIENTATION_WEIGHT * max(0, uprightness)
        + HEIGHT_GAIN_WEIGHT * height_gain                                # NEU
        - CONTROL_COST_WEIGHT   * sum(action^2)
        - JOINT_VEL_COST_WEIGHT * sum(joint_qvel^2)
```

| Konstante | Phase 1 | Phase 2 | Begründung der Änderung |
|-----------|---------|---------|--------------------------|
| `POSE_WEIGHT` | 2.0 | 2.0 | Bleibt dominanter Term — selbes Endziel |
| `POSE_SHARPNESS` | 6.0 | **3.0** | Milder — downed-Pose ist sehr weit von home, scharfe Glocke gäbe quasi 0-Gradient |
| `ALIVE_BONUS` | 0.5 | **0.1** | Niedriger — sonst lohnt sich Liegenbleiben |
| `UPRIGHT_WEIGHT` | 0.3 | **0.5** | Stärker — klares Aufwärts-Signal nötig |
| `ORIENTATION_WEIGHT` | 0.3 | **0.5** | Stärker — gleiche Begründung |
| `HEIGHT_GAIN_WEIGHT` | — | **5.0** | NEU — zieht aktiv nach oben |
| `HEIGHT_SHARPNESS` | 12.0 | 12.0 | Unverändert |
| `TARGET_HEIGHT` | 0.98 | 0.98 | Unverändert — selbe Zielhöhe |
| `CONTROL_COST_WEIGHT` | 0.001 | 0.001 | Unverändert |
| `JOINT_VEL_COST_WEIGHT` | 0.0005 | **0.0001** | Milder — Aufstehen braucht Bewegung |
| `RESET_NOISE` | 0.01 | **0.02** | Größer — variierte Start-Liegepositionen |
| `MAX_EPISODE_STEPS` | 1000 | **1500** | Länger — Aufstehen braucht Zeit |
| `DOWNED_SETTLE_STEPS` | — | **200** | NEU — Settle-Dauer für downed-Pose |

**Warum `height_gain` als positiv-only Term?** `max(0, z - z_prev)` belohnt
nur Hochkommen, bestraft aber nicht das natürliche Setzen beim Anpassen.
Ohne Clipping würde der Agent für jeden Setz-Schritt eine Strafe sehen, was
das Lernen verlangsamt.

**Reward-Balance (Begründung der Gewichte):**
- Liegen bleiben in Downed-Pose: `pose_error` groß (~0,5–1 rad² Mittel),
  `pose ≈ 2 · exp(-3 · 0,7) ≈ 0,25`; `alive 0,1`; `upright/orientation ≈ 0`;
  `height_gain = 0`. Summe ≈ 0,3–0,4 pro Step.
- Stabil stehen in home-Pose: `pose ≈ 2,0`; `alive 0,1`; `upright 0,5`;
  `orientation 0,5`; `height_gain 0` (Höhe konstant). Summe ≈ 3,1 pro Step.
- Hochkommen-Übergang: `height_gain` aktiv (`5 · Δz` pro Step) — gibt
  kontinuierliches Aufwärts-Signal während der Bewegung.

Klares globales Optimum: Hochkommen und in der home-Pose stabil bleiben.

### Observation und Action unverändert

49-dim Observation und 19-dim Action identisch zu Phase 1 — so kann
Phase-2-Training optional von einem Phase-1-Checkpoint resumen
(`train.py --resume checkpoints/h1_stand_final.zip --env standup`).
Die Phase-1-Policy weiß bereits, wie man in der home-Pose balanciert; sie
muss in Phase 2 nur das Hochkommen erlernen.

## Training-Anbindung (`train.py`)

Neues optionales argparse `--env {stand,standup}` (default `stand`). Wählt
zwischen `H1StandEnv` (Phase 1) und `H1StandupEnv` (Phase 2) via getrennter
top-level Factory-Funktionen (picklebar für `SubprocVecEnv`).

---

# Iteration 5 — Stillness-Reward (`h1_stand_env.py`)

Phase 1 lieferte nach 25M Schritten 83 % volle Episoden — der H1 steht
aufrecht. ABER: Messung über 5 Test-Episoden zeigte, dass er nicht still
steht:

| Metrik | Gemessen | Ziel |
|--------|----------|------|
| Mittlere Gelenkgeschwindigkeit | 1,71 rad/s | < 0,3 rad/s |
| xy-Drift in 10 s | 3,40 m | < 0,3 m |
| Torso-Höhen-Std | 0,019 m | < 0,01 m |

Der Roboter "wandert" während er steht. Ursache im Reward:
`JOINT_VEL_COST_WEIGHT = 0,0005` war zu mild, und es gab überhaupt keinen
Drift-Penalty.

## Neue Reward-Formel (Iteration 5)

Additive Erweiterung der Iter-2-Formel — alle bisherigen Terme bleiben mit
unveränderten Gewichten:

```
xy_drift       = sqrt(qpos[0]^2 + qpos[1]^2)        # Abstand von Startposition
root_lin_speed = sqrt(qvel[0]^2 + qvel[1]^2)        # horizontale Root-Geschw.

reward =  POSE_WEIGHT        * exp(-POSE_SHARPNESS * pose_error)
        + ALIVE_BONUS
        + UPRIGHT_WEIGHT     * exp(-HEIGHT_SHARPNESS * (z - TARGET_HEIGHT)^2)
        + ORIENTATION_WEIGHT * max(0, uprightness)
        - CONTROL_COST_WEIGHT      * sum(action^2)
        - JOINT_VEL_COST_WEIGHT    * sum(joint_qvel^2)      # 10x verschärft
        - XY_DRIFT_WEIGHT          * xy_drift               # NEU
        - ROOT_LIN_VEL_WEIGHT      * root_lin_speed         # NEU
```

## Konstanten — Änderungen vs. Iter 2

| Konstante | Iter 2 | Iter 5 | Begründung |
|-----------|--------|--------|------------|
| `JOINT_VEL_COST_WEIGHT` | 0,0005 | **0,005** | 10× verschärft. Bei gemessenen 1,71 rad/s über 19 Gelenken: `0,005 · 19 · 1,71² ≈ 0,28` Strafe/Step — klar sichtbar gegen Pose-Reward ~2,0. Bei Ziel 0,3 rad/s: `0,005 · 19 · 0,09 ≈ 0,009` — vernachlässigbar. Klare Trennung zwischen Zappeln und kleinen Korrekturen. |
| `XY_DRIFT_WEIGHT` | — | **0,5** | NEU. Linear in `\|xy\|`. Bei 1 m Drift → −0,5/Step (≈ 17 % vom Max-Reward 3,1). Bei 10 cm → −0,05 (vernachlässigbar, kleine Korrekturen erlaubt). Bei gemessenen 3,4 m → −1,7/Step, dominiert klar — Drift wird unattraktiv. |
| `ROOT_LIN_VEL_WEIGHT` | — | **0,3** | NEU. Linear in `\|v_xy\|`. Bei 0,5 m/s → −0,15/Step. Geschwindigkeitsbasiert, ergänzt den positionsbasierten Drift — verhindert auch das "Rennen-und-Stoppen", das einen reinen Positions-Penalty austricksen könnte. |

Alle anderen Iter-2-Gewichte (`POSE_WEIGHT 2.0`, `ALIVE_BONUS 0.5`,
`UPRIGHT_WEIGHT 0.3`, `ORIENTATION_WEIGHT 0.3`, `CONTROL_COST_WEIGHT 0.001`)
und Schärfen (`POSE_SHARPNESS 6.0`, `HEIGHT_SHARPNESS 12.0`) unverändert.
Termination unverändert (`FALL_HEIGHT 0.5`), Observation-Space (49), Action
(19), `FRAME_SKIP` unverändert.

## Reward-Balance — globales Optimum: Statue an x=y=0

- **Ruhige Statue in home-Pose bei x=y=0**: `pose ≈ 2,0`; `alive 0,5`;
  `upright 0,3`; `orientation 0,3`; alle Penalties ≈ 0 → **Reward ≈ 3,1/Step**.
- **Aktueller Zustand (Iter 4-Policy gemessen)**: pose-Term gut (~1,9),
  aber `joint_vel_cost ≈ 0,28`, `xy_drift_cost ≈ 1,7`,
  `root_lin_vel_cost ≈ 0,1`. Netto-Reward sinkt deutlich — Policy-Gradient
  zeigt jetzt klar Richtung Stillstand.
- **Kleine Korrekturen** (10 cm Drift, 0,3 rad/s, 0,1 m/s): Penalties
  zusammen ≈ 0,01 + 0,05 + 0,03 = 0,09/Step. Vernachlässigbar — Roboter
  darf weiterhin korrigieren, um nicht zu kippen.

Alle Stillness-Penalties sind **linear** (nicht quadratisch in
Position/Geschwindigkeit). Das gibt einen konstanten Lernsignal-Gradienten
auch bei großem Drift, statt eines bei Null verschwindenden Gradienten wie
bei Quadrat-Penalties.

## `info`-Dict-Erweiterung

`step()` liefert zusätzlich `xy_drift` (m), `xy_drift_cost`,
`root_lin_vel_cost` — für TensorBoard und Evaluation.

---

# Iteration 6 — Anatomisch differenzierter Reward + Foot-Lock

Iter 5 schaffte zwar einen Drift-Penalty (Pelvis-xy), aber das uniforme
`JOINT_VEL_COST_WEIGHT = 0,005` zwang die Policy in ein Kompromiss-Optimum:
Wer die Beine ruhig hält, muss auch die Arme starr machen — und verliert
damit den Balance-Trick. Resultat war "aktive Wackelbalance".

Insight-Bericht: `insights/01_stillstand.md`. Kernidee: **anatomisch
differenzieren** — der Mensch hält Stillstand mit ruhigen Beinen UND
schwingenden Armen (Drehimpuls-Balance). Reward muss das spiegeln.

## Zwei chirurgische Eingriffe

**1. Differenzierter Joint-Velocity-Penalty (Lower vs. Upper)**

Statt eines uniformen Penalty wird `qvel[6:25]` in zwei Gruppen geteilt:

| Gruppe | Aktuator-Indizes | Gelenke | Gewicht | Funktion |
|--------|------------------|---------|---------|----------|
| LOWER | 0–9 | Hip-Yaw/Roll/Pitch, Knee, Ankle (beide Seiten) | **0,020** | "unten fest" — Standfläche-Stabilität |
| UPPER | 10–18 | Torso, Shoulder-Pitch/Roll/Yaw, Elbow (beide Seiten) | **0,0005** | "oben balance" — Arme dürfen schwingen |

Faktor 40× zwischen Gruppen. Damit ist Bein-Zappeln teuer, Arm-Schwung billig.

**2. Foot-Position-Lock + Foot-Air-Penalty**

Pelvis-xy ist die falsche Messgröße. Olafs Heuristik meint die **Füße** —
der Kontaktpunkt zur Welt. Reset speichert
`self._foot_xy_init[i]` für `i ∈ {left, right}` (jeweiliger
Body-`xpos[:2]` nach Noise-Anwendung). Pro Step:

```
left_foot_drift  = ‖xpos[left_ankle_link, :2]  - foot_xy_init[0]‖
right_foot_drift = ‖xpos[right_ankle_link, :2] - foot_xy_init[1]‖
foot_lock_cost   = FOOT_POS_LOCK_WEIGHT * (left_foot_drift + right_foot_drift)
```

Zusätzlich pro Fuß **ohne Bodenkontakt** ein Strafsatz:

```
foot_air_cost = FOOT_AIR_PENALTY * (foot_air_left + foot_air_right)
```

Foot-Contact wird über die `data.contact[:ncon]`-Liste ermittelt:
ein Fuß gilt als am Boden, sobald ein Contact-Eintrag entweder `geom1`
oder `geom2` zu seinem Body gehört.

## Neue Reward-Formel (Iteration 6)

```
joint_qvel = qvel[6:25]
lower_vel_cost = LOWER_JOINT_VEL_WEIGHT * sum(joint_qvel[LOWER_JOINT_IDX]^2)
upper_vel_cost = UPPER_JOINT_VEL_WEIGHT * sum(joint_qvel[UPPER_JOINT_IDX]^2)
foot_lock_cost = FOOT_POS_LOCK_WEIGHT  * (left_foot_drift + right_foot_drift)
foot_air_cost  = FOOT_AIR_PENALTY      * (foot_air_left + foot_air_right)
root_lin_vel_cost = ROOT_LIN_VEL_WEIGHT * sqrt(qvel[0]^2 + qvel[1]^2)

reward =  POSE_WEIGHT        * exp(-POSE_SHARPNESS * pose_error)
        + ALIVE_BONUS
        + UPRIGHT_WEIGHT     * exp(-HEIGHT_SHARPNESS * (z - TARGET_HEIGHT)^2)
        + ORIENTATION_WEIGHT * max(0, uprightness)
        - CONTROL_COST_WEIGHT * sum(action^2)
        - lower_vel_cost - upper_vel_cost
        - foot_lock_cost - foot_air_cost
        - root_lin_vel_cost
```

## Konstanten — Änderungen vs. Iter 5

| Konstante | Iter 5 | Iter 6 | Begründung |
|-----------|--------|--------|------------|
| `JOINT_VEL_COST_WEIGHT` | 0,005 | **entfernt** | Uniform-Penalty zwang Arm-Starre — siehe Insight-Bericht §3. |
| `LOWER_JOINT_VEL_WEIGHT` | — | **0,020** | 4× alt. Bei Bein-Zappeln (1,5 rad/s × 10) ≈ 0,45/Step — dominant. Bei Ziel (0,2 rad/s) ≈ 0,008 — erlaubt. |
| `UPPER_JOINT_VEL_WEIGHT` | — | **0,0005** | 10× milder als Iter 5. Bei Arm-Schwung (1,0 rad/s × 9) ≈ 0,005 — vernachlässigbar. Drehimpuls-Balance bleibt zugänglich. |
| `XY_DRIFT_WEIGHT` | 0,5 | **entfernt** | Pelvis-Drift redundant zu Foot-Lock; Foot-Lock präziser (misst Kontaktpunkt). |
| `FOOT_POS_LOCK_WEIGHT` | — | **2,0** | Linear. 5 cm/Fuß → −0,2 (Korrektur erlaubt). 30 cm → −1,2 (klar bestraft). 1 m Wander-Modus → −4,0/Step (dominant). |
| `FOOT_AIR_PENALTY` | — | **0,5** | Pro Fuß in der Luft. Ein Fuß abgehoben → −0,5. Spürbar, aber weich genug für PPO-Lernsignal. |
| `ROOT_LIN_VEL_WEIGHT` | 0,3 | **0,3** | Behalten. Geschwindigkeitsbasiert ergänzend zum positionsbasierten Foot-Lock. |

Alle übrigen Iter-2-Terme unverändert (`POSE_WEIGHT 2,0`, `POSE_SHARPNESS 6,0`,
`ALIVE_BONUS 0,5`, `UPRIGHT_WEIGHT 0,3`, `ORIENTATION_WEIGHT 0,3`,
`CONTROL_COST_WEIGHT 0,001`, `HEIGHT_SHARPNESS 12,0`).
Termination, Observation (49), Action (19), `FRAME_SKIP` unverändert.

## Reward-Balance — globales Optimum: Statue mit lebenden Armen

**Ruhige Statue** (Füße fest, Beine still, Arme dürfen schwingen):
- pose 2,0 + alive 0,5 + upright 0,3 + orientation 0,3 = **3,1**
- Penalties: lower_vel ≈ 0,008, upper_vel ≈ 0,005, foot_lock ≈ 0,
  foot_air ≈ 0, root_lin ≈ 0,03, control ≈ 0 → Summe ≈ 0,05
- **Netto ≈ 3,05/Step** — globales Maximum

**Aktuelle Wander-Policy** (gemessen):
- pose ≈ 1,9 + alive 0,5 + upright 0,3 + orientation 0,3 = 3,0
- Penalties: lower_vel ≈ 0,45, upper_vel ≈ 0,02, foot_lock ≈ 4,0,
  foot_air ≈ 0, root_lin ≈ 0,1
- **Netto ≈ −1,6/Step** — klar negativ, Policy-Gradient zeigt scharf weg

**Linearität:** Foot-Lock + Root-Lin-Vel weiterhin linear, nicht quadratisch.
Konstanter Gradient auch bei großem Drift.

## Verwendete Body-Namen (verifiziert im Modell)

| Zweck | Body-Name | Body-ID |
|-------|-----------|---------|
| linker Fuß | `left_ankle_link` | 6 |
| rechter Fuß | `right_ankle_link` | 11 |

Die Foot-Geometries (foot1/foot2/foot3 capsules) sind diesen Bodies
zugeordnet — daher Foot-Contact-Detektion über deren Body-ID korrekt.

## `info`-Dict-Erweiterung

`step()` liefert zusätzlich: `lower_vel_cost`, `upper_vel_cost`,
`foot_lock_cost`, `foot_air_cost`, `foot_air_left`, `foot_air_right`.
Iter-5-Felder `joint_vel_cost`, `xy_drift`, `xy_drift_cost` entfallen.
