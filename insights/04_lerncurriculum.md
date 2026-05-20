# Lern-Curriculum nach menschlichem Vorbild — H1 RL

**Bericht 4 der Domain-Insight-Reihe** | Quelle: der Insight (2026-05-20)

---

## 1. Kurzfassung

Das aktuelle PPO-Training wirft den H1 in Phase 6 — freies Stehen — ohne je die Vorstufen gelernt zu haben. Menschen brauchen ~12 Monate, um genau diese 6 Phasen sequenziell aufzubauen. Der H1 versucht das in einem Lauf zu erzwingen. Jede Phase erschließt einen anderen mechanischen Hebel: Phase 2 baut Körpergefühl für Bodenkontakt, Phase 3–4 trainieren Pose-Halten ohne Balance-Last, Phase 5 gibt der Policy Spielraum bevor der Knöchel alleine arbeiten muss. Der Knöchel-82%-Befund (Bericht 03) zeigt direkt warum Phase 5 der kritischste fehlende Schritt ist — er verschafft Spielraum bevor volle Auslastung verlangt wird.

---

## 2. Die 6 Phasen — technisch übersetzt

| Phase | Menschlich | H1-Realisierung in MuJoCo | Env-Datei | Reward-Schwerpunkt | Erwartetes Resultat |
|-------|-----------|--------------------------|-----------|-------------------|-------------------|
| **1 — Liegen** | Neugeborenes auf Rücken/Bauch | Reset-Pose = `downed` (existiert in `h1_standup_env.py`). Keine Bewegung gefordert. | `h1_standup_env.py` (existing) | Alive-Bonus, Orientierung-Torso messen | Keine Policy nötig — Baseline-Zustand |
| **2 — Krabbeln** | 4 Stützpunkte, Vortrieb in x | Knie + Handgelenke am Boden, `geom_bodyid`-Kontakte für alle 4 Punkte prüfen. Reward: x-Position-Gewinn + Kontakt-Bonus je Stützpunkt. **Impl-Option:** Kontakt-Check auf `knee_link` + `wrist_link` via `data.contact[]` | `h1_crawl_env.py` (neu) | `+x_pos_gain * 0.5`, `+contact_bonus * 4` (je 0.25), `-feet_off_ground` | Policy lernt Boden-Druck + Koordination vor Gewichtsübernahme |
| **3 — Aufstehen mit Hilfe** | Hochziehen an Möbel/Eltern-Arm | **Virtuell:** `xfrc_applied[torso_id, :3]` — Aufwärtskraft am Torso, proportional zur Höhe unter Ziel-Stand-Höhe. Reale Alternative: Box-Geom als Geländer + Kollision. Virtuelle Variante einfacher, kein Greif-Mechanismus nötig. | `h1_standup_assisted_env.py` (neu) | `+height_gain`, `-contact_lost` (Stütz-Kraft abschalten wenn Höhe = Stand) | Policy lernt Aufsteh-Bewegung ohne Balance-Pflicht |
| **4 — Ruhiges Anhalten** | Stehen, Hände an Möbel | Stütz-Kraft bleibt (schwächer als Phase 3), `xfrc_applied` am Torso = 20% des Körpergewichts. Reward: Pose-Tracking + Höhe halten. | `h1_standup_assisted_env.py` + Flag `assist_level=0.2` | `+pose_match * 2.0`, `+height_hold * 0.5`, `-joint_vel_lower * 0.01` | Pose-Halten ohne volle Knöchel-Last — Basis für Phase 5 |
| **5 — Stehen mit Balance-Hilfe** | Eltern halten Hände — Kraft nimmt ab | Stütz-Kraft am Torso startet bei 30% Körpergewicht, linearer Abfall auf 0 über Episoden-Verlauf (`assist = max(0, 0.3 - step/max_steps)`). Kein Geländer. | `h1_stand_env.py` + Flag `--balance-assist` | Reward = Phase 6 identisch, aber Knöchel-Reserve durch Assist-Entlastung | Policy lernt echte Balance mit Sicherheitsnetz — **Schlüsselphase** |
| **6 — Freies Stehen** | Kein Halt, kein Helfer | Aktuelles `h1_stand_env.py`, Iter 5/6 Stand | `h1_stand_env.py` (existing) | Alle Rewards aus Berichten 01 + 03 | Was wir aktuell trainieren (90% Erfolg aber Knöchel-Grenze) |

**Implementierungs-Detail Phase 3–5 (xfrc_applied):**
```python
# Torso-ID im Reset einmal abrufen
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

# In step(): externe Aufwärtskraft anwenden
assist_force = assist_level * model.body_mass[torso_id] * 9.81  # Anteil Körpergewicht
data.xfrc_applied[torso_id, 2] = assist_force  # z-Achse = nach oben
```

---

## 3. Welche Phase hilft dem aktuellen Projekt am meisten?

**Empfehlung: Phase 5 — Stehen mit Balance-Hilfe.**

Begründung:

| Befund | Verbindung zu Phase 5 |
|--------|----------------------|
| Knöchel bei **82 % Auslastung** allein fürs Halten (Bericht 03) | Assist-Kraft am Torso reduziert effektiven CoM-Offset — Knöchel-Halte-Drehmoment sinkt von 33 Nm auf ~23 Nm bei 30% Assist. Das sind **17 Nm Spielraum** für dynamische Korrekturen. |
| qvel_mean = 1,11 rad/s nach 35M Schritten | Policy zappelt weil Knöchel gegen Grenze drückt (Ankle-Strategy scheitert an 82% Limit). Assist verschafft Spielraum für echte Ankle-Strategy. |
| xy-Drift 0,53 m | Drift entsteht weil Policy bei Knöchel-Sättigung auf Schritt-Strategie ausweicht. Phase 5 verhindert Sättigung. |
| Phase 6 direkt = „alles auf einmal" | Phase 5 isoliert Balance-Lernen von Halte-Lernen — Policy muss nicht gleichzeitig Schwerkraft kompensieren UND Balance finden. |

**Timing:** Resume vom Iter-6-Endcheckpoint mit `assist=0.3`, schrittweise auf 0 reduzieren über 5–10M Steps. Kein Neustart nötig.

---

## 4. Was kostet das, was bringt es?

| Dimension | Einschätzung |
|-----------|-------------|
| **Aufwand** | Phase 5 als Flag in bestehendem `h1_stand_env.py`: ~30 Zeilen. Vollständiges 6-Phasen-Curriculum mit Curriculum-Manager: neue `curriculum.py` + Schalt-Logik + je 1 Env-Datei = ~3–4 Tage |
| **Risiko: Schummeloptima** | Jede Phase kann eine eigene stabile Lösung finden, die in der nächsten Phase nicht hilft. Phase 2 (Krabbeln) könnte z.B. lernen, sich mit Armen zu schieben statt Beinen. Mitigation: kurze Phasen-Dauer (5–10M Steps), Policy-Transfer mit gefrorenen Layern. |
| **Nutzen** | Policy lernt sukzessiv: erst Bodenkontakt → dann Höhe → dann Pose → dann Balance. Jede Phase konvergiert schneller weil das Lernproblem kleiner ist. Knöchel-Reserve wird systematisch aufgebaut statt sofort gefordert. |
| **Tradeoff Zeit** | 25M-Lauf × 6 Phasen ≈ 150M Schritte gesamt. ABER: jede Phase konvergiert in 5–15M — realistisch 60–80M Schritte total. Nicht 6× länger, eher 2–3×. |
| **Größter Gewinn** | Phase 5 allein, als Resume-Add-on, kostet 10M Steps extra und gibt voraussichtlich den Durchbruch — ohne vollständiges Curriculum zu bauen. |

---

## 5. Minimal-Viable nächster Schritt

**Kein Pflicht-Vorschlag — nur ein konkreter Test-Einstieg.**

Phase 5 als Flag `--balance-assist` in das bestehende `h1_stand_env.py` einfügen. Resume vom Iter-6-Checkpoint mit assist stark, linear auf 0 reduzieren.

```python
# In h1_stand_env.py — ergänzen, nichts löschen

class H1StandEnv(gymnasium.Env):
    def __init__(self, ..., balance_assist: float = 0.0):
        ...
        self._balance_assist_init = balance_assist   # 0.0 = aus, 0.3 = Phase 5
        self._step_count = 0
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )

    def reset(self, **kwargs):
        self._step_count = 0
        self.data.xfrc_applied[:] = 0.0          # externe Kräfte zurücksetzen
        return super().reset(**kwargs)

    def step(self, action):
        # Assist-Kraft berechnen (linear abnehmend)
        progress = min(1.0, self._step_count / 5_000_000)
        current_assist = self._balance_assist_init * (1.0 - progress)

        # Aufwärtskraft am Torso anwenden (vor mj_step)
        torso_mass = self.model.body_mass[self._torso_id]
        self.data.xfrc_applied[self._torso_id, 2] = current_assist * torso_mass * 9.81

        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1
        info["balance_assist"] = current_assist
        return obs, reward, terminated, truncated, info
```

**Starten:**
```bash
python train.py --resume iter6_checkpoint --balance-assist 0.3 --total-steps 10_000_000
```

**Was beobachten:** `balance_assist` in TensorBoard loggen. Wenn qvel_mean unter 0.5 rad/s fällt während assist noch > 0.1, ist Phase 5 am Lernen. Wenn Policy kollabiert wenn assist = 0: Phase 4 erst durchlaufen.

---

## 6. Verknüpfung zu früheren Berichten

### Bericht 01 — Stillstand / Lower-Upper-Split

| Bezug | Wirkung wenn Phase 5 verbessert |
|-------|--------------------------------|
| `qvel_mean` (Ziel < 0.3 rad/s) | Knöchel-Spielraum durch Assist → Lower-Gruppe kann ruhiger werden |
| `xy_drift` (Ziel < 0.1 m) | Policy weicht nicht mehr auf Schritt-Strategie aus → Drift sinkt direkt |
| Foot-Lock-Penalty | Weniger Knöchel-Sättigung = Fuß bleibt ruhiger = geringerer Penalty = Reward steigt |
| Ankle-Strategy (< 0.2 rad/s Mikro-Korrektur) | Phase 5 gibt Raum genau für diesen Mechanismus — ist das Ziel |

### Bericht 03 — Aktor-Einstellungen

| Bezug | Wirkung wenn Phase 5 verbessert |
|-------|--------------------------------|
| **Knöchel 82 % Auslastung** | Assist-Kraft 30% → Halte-Drehmoment sinkt von 33 Nm auf ~23 Nm → Auslastung ~57 % → 17 Nm Reserve für Korrekturen |
| Action-Anchor-Term (Vorschlag Bericht 03) | Phase-5-Policy landet näher an Static-Torque-Werten → Anchor-Term wird effizienter wenn er später dazukommt |
| Hüft-Roll 18 % Auslastung | Unverändert — nicht der Engpass. Bleibt Beobachtungsgröße. |

**Schlüssel-Metrik für Erfolg von Phase 5:**
- Knöchel-Drehmoment-Varianz sinkt (beobachtbar via `data.actuator_force[4]` und `[9]` in TensorBoard)
- `qvel_mean` LOWER-Gruppe unter 0.4 rad/s während assist noch läuft
- Nach assist=0: kein Einbruch, sondern stabiles Niveau halten

---

**Karpathy-Sanity:**
- Trace ✅ — jede Phase direkt auf der Insight-Wortlaut rückführbar
- Senior-Check ✅ — Phase 5 als Flag ist 30 Zeilen, nicht 300
- Knöchel-82%-Bezug ✅ in Sektion 3 + 6 quantifiziert
- Alle 6 Phasen in Tabelle ✅
- Konkrete MuJoCo-Mechanismen ✅ (xfrc_applied, data.contact[], geom-basiert)
