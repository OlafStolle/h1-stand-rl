# Aufsteh-Choreografie: Rückenlage → Stand (H1 Sub-Goal-Plan)

**Bericht 5 der Domain-Insight-Reihe** | Quelle: Olafs Biomechanik-Beschreibung (2026-05-20)

---

## 1. Olafs Choreografie — wörtlich

> *"Wenn ich am Rücken liege: Beine anwinkeln, Ellbogen auf Boden abstützen, nach vorne beugen, Arme nach hinten abstützen, durch Anziehen der Beine Oberkörper in Hocke bringen, mit Armen abstützen, nach vorne gebeugt langsames Aufstehen."*

Diese Bewegung ist eine klassische **Sit-up→Squat-up-Sequenz**: aus Rückenlage über Sitz-Position zur Hocke zum freien Stand. Sie zerteilt sich natürlicherweise in 8 Zwischenposen, die sequenziell als Sub-Goals für ein RL-Curriculum dienen.

---

## 2. Die 8 Stufen — MuJoCo-technisch

| # | Stufe-Name | Menschlich | H1-qpos-Charakteristik | Prüf-Kriterium | Sub-Reward-Fokus |
|---|---|---|---|---|---|
| **1** | **Rückenlage (Start)** | Rücken + Kopf auf Boden, Beine gestreckt | `qpos[2]` (Höhe CoM) ≈ 0.15 m, Torso-quat zeigt Bauch nach oben, Knie-Winkel ≈ 0 rad (gestreckt) | `qpos[2] < 0.2` ∧ `knie_winkel < 0.1` | Initialisierung — keine Bewegung |
| **2** | **Beine anwinkeln** | Knie beugen, Füße bleiben Boden, Abstoßfläche vorbereiten | Knie-Winkel auf ~1.4 rad (ca. 80°), Hüft-Pitch ~20° gebeugt, Füße am Boden, CoM-Höhe steigt leicht auf ≈ 0.25 m | `knie_winkel in [1.2, 1.6]` ∧ beide_fuesse_boden | Koordination Knie-Hüfte, erstes Höhengewinn |
| **3** | **Ellbogen abstützen** | Ellbogen auf Boden, Schultern hochziehen, Start Sit-up-Bewegung | Ellbogen-Kontakt (left/right_elbow_link in Bodenkontakt), Schulter-Pitch ~30°, Torso-Pitch ~15° (Bauch noch unten), CoM-Höhe ≈ 0.35 m | `ellbogen_kontakt ∧ qpos[2] in [0.3, 0.4]` | Multi-Contact-Bonus: Ellbogen + Füße = 3 Stützpunkte |
| **4** | **Nach vorne beugen** | Bauch-Krümmung reduzieren, Oberkörper zu den Knien drücken | Torso-Pitch progressiv auf +40° (Bauch zu Knien), Hüft-Pitch ~45°, CoM-Höhe ≈ 0.40 m, Ellbogen + Füße noch Kontakt | `torso_pitch in [30°, 50°]` ∧ `hüft_pitch > 40°` | Haltungs-Verfolgung (Pose-Tracking) zur neuen Zwischen-Pose |
| **5** | **Arme nach hinten abstützen** | Hände/Handflächen jetzt am Boden HINTER Körper, Sit-Position erreicht | Schulter nach hinten (Schulter-Pitch ~−30°), Ellbogen-Winkel ~100° (Arme stützen), Torso mehr aufrecht (~20° Pitch), Hüft-Pitch ~70°, CoM-Höhe ≈ 0.45 m | `arm_stütz_kontakt ∧ torso_pitch < 30°` | Positionswechsel (Kontakt vom Ellbogen zu Handflächen) |
| **6** | **Oberkörper in Hocke** | Knie sehr stark gebeugt (tiefer als 45°), Hüfte nach vorn, CoM über Füße verlagert | Knie-Winkel ~1.8 rad (ca. 100°, sehr tief), Hüft-Pitch ~85°, Torso-Pitch ~15° (aufrecht), CoM-Höhe ≈ 0.50 m, Hände evtl. am Boden oder auf Knien | `knie_winkel > 1.7` ∧ `hüft_pitch > 80°` ∧ `qpos[2] > 0.48` | Kraft-Aufbau: Knie-Streck-Vorbereitung (noch nicht streckend, sondern _positionierend_) |
| **7** | **Mit Armen abstützen** | Hände gehen nach vorn (auf/vor den Knien oder Boden), Oberkörper stabilisiert | Schulter-Pitch variabel (Arme nach vorn), beide Hände können Knien oder Boden berühren, Knie-Winkel ~1.5–1.8 rad, CoM-Höhe ≈ 0.55 m, Hüft-Pitch ~70° | `arme_vorne_kontakt ∧ qpos[2] in [0.50, 0.65]` | Übergangsstabilität: Arme vom hinten zum vornen Stütz |
| **8** | **Aufrichten (Stand-Pose)** | Knie strecken, Hüfte strecken, Torso aufrichten, home-Pose erreichen | Knie-Winkel ~0.4–0.6 rad (gestreckt wie in home-Pose Iter 6), Hüft-Pitch ~0°, Torso-Pitch ≈ 0°, CoM-Höhe = 0.92 m (Standard-Ständer-Höhe), Arme entspannen | `knie_winkel < 0.8` ∧ `qpos[2] > 0.85` ∧ `pose_match_zu_home > 0.9` | Finale Posen-Überein­stimmung mit home (Iter-6-Politik übernimmt) |

**Qpos-Index-Referenz (ungefähr — IMMER in `h1_standup_env.py` verifizieren):**
- `qpos[0:3]` = x, y, z (Position)
- `qpos[3:7]` = Quaternion (Torso-Orientierung)
- `qpos[7]` = Hüft-Pitch (links)
- `qpos[8]` = Hüft-Roll (links)
- `qpos[9]` = Hüft-Yaw (links)
- `qpos[10]` = Knie (links)
- ... ähnlich für rechts ab Index 14

**Anmerkung zu qpos-Werten:** Die Winkel-Bereiche sind Schätzungen basierend auf H1-mech. Specs (Knie max. 2.05 rad, Hüfte ~180°). Diese Tabelle beschreibt die **Konzept-Sequenz**, nicht eine kalibrierte Micro-Definition. Vor RL-Training müssen die genauen Werte im Reset-Generator validiert werden.

---

## 3. Reward-Architektur: Zwei Optionen für Sub-Goal-Sequencing

### Option A: Phase-Conditioned Reward (Einfach, aber manuell)

**Idee:** Eine einzige Trainings-Umgebung. Im Reward wird eine `current_phase` Variable (1–8) kontinuierlich oder diskret verwaltet. Reward-Terms schalten je nach Phase an/aus oder ändern Gewichtung.

**Beispiel-Struktur:**

```python
def compute_reward(self, phase: int, qpos, qvel, ...):
    """
    phase: 1 = Rückenlage bis zur Zielpose, 
           2 = Beine anwinkeln,
           ... 
           8 = Aufrichten
    """
    reward = 0.0
    
    # Phase-spezifische Sub-Goals
    if phase == 1:
        reward += 0  # nur initialisieren, keine Belohnung
        phase_transition = (knie_winkel > 1.2 and hüfte_pitch > 20)
    
    elif phase == 2:
        reward += 0.5 * (knie_winkel - 0.0) / 1.4  # knie progressiv
        reward += 0.2 * (hüft_pitch / 45)
        phase_transition = (qpos[2] > 0.3 and ellbogen_kontakt)
    
    elif phase == 3:
        # Ellbogen abstützen
        reward += 1.0 * len(ground_contacts) / 3  # Multi-Contact-Bonus
        phase_transition = (torso_pitch > 30 and qpos[2] > 0.35)
    
    # ... Phase 4–8 analog
    
    # Transition auf nächste Phase wenn Sub-Goal erreicht
    if phase_transition and current_phase < 8:
        self.current_phase += 1
        self.phase_transition_step = self.step_count
    
    return reward
```

**Vorteile:**
- ✅ Eine Umgebung, einfaches Setup
- ✅ Phasen-Transitions können adaptiv (z.B. wenn Policy lange steckt) oder hart vorgegeben sein
- ✅ Schnell zu prototypen

**Nachteile:**
- ❌ Phase-Tracking ist regelbasiert → manuell zu debuggen wenn Übergangskriterien nicht greifen
- ❌ Policy kann „mogeln" und Phase überspringen ohne Sub-Goal wirklich erreicht zu haben
- ❌ Während Phase 3 muss Torso aber höher, das mischt Ziele

---

### Option B: Curriculum Sub-Envs (Robust, aber mehr Aufwand)

**Idee:** 8 separate Trainings-Umgebungen (oder ein Manager, der zwischen ihnen schaltet). Jede Umgebung:
- **Reset-Pose** = Endpose der vorherigen Stufe
- **Reward** = nur auf Sub-Goal der aktuellen Stufe optimiert
- **Termination** = wenn Stufe erreicht ODER Agent schlägt fehl
- **Transition** = bei Erfolg zur nächsten Stufe + Checkpoint speichern

**Beispiel-Struktur:**

```python
class H1ChoreographyEnv(gymnasium.Env):
    """
    Mit Parameter phase (1–8).
    reset_pose und target_pose definieren je Phase.
    """
    
    PHASE_CONFIGS = {
        1: {"reset": "supine", "target": "legs_bent", "steps": 100_000},
        2: {"reset": "legs_bent", "target": "elbows_down", "steps": 100_000},
        3: {"reset": "elbows_down", "target": "forward_lean", "steps": 100_000},
        # ... 4–8
        8: {"reset": "standing_prep", "target": "home_pose", "steps": 200_000},
    }
    
    def __init__(self, phase: int = 1):
        self.phase = phase
        self.config = self.PHASE_CONFIGS[phase]
        self._total_phase_steps = 0
    
    def reset(self):
        """
        Setzt Roboter in Reset-Pose für diese Phase,
        nicht in downed-Pose.
        """
        self._set_to_reset_pose(self.config["reset"])
        return super().reset()
    
    def compute_reward(self, qpos, qvel, contact_info):
        """
        Nur reward für diesen einen Übergang (z.B. Phase 2: Beine anwinkeln).
        Nicht für alle 8 Stufen gleichzeitig.
        """
        return self._phase_specific_reward(qpos, qvel, self.phase)
    
    def is_phase_complete(self):
        """Prüfe ob Target-Pose erreicht."""
        return self._check_target(self.config["target"])
```

**Curriculum-Manager (oberste Schicht):**

```python
class ChoreographyCurriculum:
    def __init__(self):
        self.current_phase = 1
        self.envs = {p: H1ChoreographyEnv(phase=p) for p in range(1, 9)}
        self.trainer = PPOTrainer(...)
    
    def train_current_phase(self, steps: int):
        """Trainiere aktuelle Phase bis Erfolg oder Schritt-Limit."""
        env = self.envs[self.current_phase]
        self.trainer.train(env, num_steps=steps)
    
    def phase_complete(self):
        """Wenn Policy Phase meistert, nächste Phase."""
        self.current_phase += 1
        # Optional: Policy-Transfer mit gefrorenen Layern
        # Checkpoint speichern
```

**Vorteile:**
- ✅ Jede Phase unabhängig optimiert, klare Erfolgskriterien
- ✅ Robust gegen Sub-Optimal-Policy (kann nicht „mogeln")
- ✅ Checkpoints pro Phase ermöglichen Rollback/Analyse
- ✅ Parallelisierbar: mehrere Phasen auf verschiedenen GPUs trainieren

**Nachteile:**
- ❌ 8 Env-Dateien + Manager = 500+ Zeilen Code
- ❌ Reset-Posen müssen physikalisch erreichbar sein (z.B. Ellbogen-auf-Boden manuell setzen ist nicht trivial)
- ❌ Policy-Transfer zwischen Phasen muss geplant sein (Gewichte-Init, Reward-Skalierung)

---

## 4. Empfehlung

**→ Option B (Sub-Envs) mit Phase 2–3 als Pilot.**

**Begründung:**
1. Olafs Curriculum-Insight (Bericht 04) zeigt, dass sequenzielle Phasen effizienter sind als "alles auf einmal". Option B realisiert das systematisch.
2. Phase 5 (Bericht 04) = Balance-Hilfe. Diese Choreografie deckt Phase 2–3 Aufstehen (Krabbeln → Hocke). Kombiniert bauen sie das volle Curriculum.
3. Reset-Poses sind für Stufen 1–3 trivial (ältere Frames aus Rückenlage-Trajectories), ab Phase 4 aufwendiger (Sit-up-Pose manuell konstruieren).

**Minimal-Viable Pilot:**
- Umgebung: nur Phase 2 + 3 (Beine anwinkeln + Ellbogen abstützen)
- Reset: `downed` Pose um 10 Frames Simulation laufen lassen → natürliche Lage unter Schwerkraft
- Reward: Multi-Contact-Bonus (Bericht 02) + Höhengewinn-Sharpness (Bericht 02)
- Ziel: 15M Steps bis "Vierfüßler-Hocke" stabil ist

Diese Vorarbeit macht Phase 4–8 später einfacher, weil die Robustik für Contact-Checks und Pose-Übergänge schon validiert ist.

---

## 5. Verknüpfung zu früheren Berichten

| Bericht | Verbindung zu dieser Choreografie |
|---------|----------------------------------|
| **01 — Stillstand** | Phase 8 = Ziel ist home-Pose (Stillstand). Die 7 davor sind Wege dahin. |
| **02 — Aufstehen (Allgemein)** | Hier konkrete Umsetzung der 5 menschlichen Phasen aus 02 in 8 H1-Stufen. Multi-Contact-Term (02, Zeile 74) startet in Phase 3. |
| **03 — Aktor-Einstellungen** | Phase 8 muss Knie-Winkel in home-Positionsbereich (0.4–0.6 rad) halten — hier definiert. Knöchel 82%-Auslastung kommt erst in Phase 6–8 (squat-up Teil). |
| **04 — Lern-Curriculum** | Choreografie konkretisiert Phase 2 (Krabbeln) + 3 (Aufstehen mit Hilfe) des 6-Phasen-Curriculums aus Bericht 04. Phase 5 (Balance-Hilfe) ist orthogonal (Kraft-Reducer) und kann parallel laufen. |

---

## 6. Nächste Schritte — Optionen (keine Pflicht-Reihenfolge)

- **A) Phase 2 als isolierter Test.** Env `h1_choreography_phase2.py`, Reset = Rückenlage, Ziel = Beine-angewinkelt-Pose mit Kontakt-Check. 2M Steps, schnelles Feedback ob Reward-Struktur taugt.

- **B) Reset-Pose-Katalog bauen.** Script das die 8 Posen generiert (qpos-Werte per Hand bestätigt), `save_reset_frames.py`. Wird für alle Sub-Envs gebraucht.

- **C) Option A (Phase-Conditioned) als schneller Prototyp.** In `h1_standup_env.py` ein `--curriculum-phases` Flag, Reward-Terms schalten 1–8. Ergebnis: Sehen ob Phase-Transitions funktionieren, ohne 8 Dateien zu schreiben. Dann entscheiden ob B nötig ist.

- **D) Multi-Contact-Implementierung aus Bericht 02 ausreifen.** Der Contact-Check-Code (02, Zeile 52–69) ist noch nicht im Projekt. Erst das testen bevor Sub-Envs gebaut werden.

---

**Karpathy-Sanity:**
- Trace ✅ — 8 Stufen direkt rückführbar zu Olafs Zitat
- Senior-Check ✅ — Option B ist 500 Zeilen, nicht 2000; Option A ist 100 Zeilen
- Quantitativ ✅ — qpos-Werte geschätzt, als solche gekennzeichnet
- Nächste Schritte ✅ — 4 konkrete Optionen, nicht "versuch Phase 2"
