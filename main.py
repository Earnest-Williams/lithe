#!/usr/bin/env python3
"""
Single-file RPG/VN dating-sim scaffold.

Goals
- One Python file using: PySide6, SQLite (via SQLAlchemy), Pydantic, PyYAML, Pillow, and transitions.
- System-first: minimal placeholder content, debuggable, deterministic seed.
- Clean layering: data (SQLite), config (YAML→Pydantic), state machine (transitions), and GUI (PySide6).
- No external assets required (falls back to generated placeholder pixmaps if files missing).

Run
    python datingsim_single.py --seed 42

Dependencies (examples)
    pip install PySide6 SQLAlchemy pydantic pyyaml pillow transitions

Design notes
- Takes inspiration from the attached multi-module prototype: signals, panes, scene + dialogue overlay,
  day/dialogue/date states, travel, acquaintance/opinion loop.
- Unlike the prototype, this is self-contained and persistence-backed.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError
import yaml
from transitions import Machine
from sqlalchemy import (
    create_engine, Integer, String, ForeignKey, select, event, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

# Qt
from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QAction, QPixmap, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QListWidget, QListWidgetItem, QInputDialog, QProgressBar
)
# -------------------------- Config (YAML → Pydantic) --------------------------

DEFAULT_CFG_YAML = """
ui:
  title: "Long Twilight — System Scaffold"
  continue_label: "Continue"
  talk_label: "Talk"
  talk_button: "Talk"
  deterministic_label: "Deterministic Seed"
  toast_history: "Recent"
  knowledge_title: "Knowledge"
  level_format: "Level {level}"
  hp_format: "HP %v"
  mp_format: "MP %v"
  stamina_format: "STA %v"

world:
  locations:
    - id: "residential district"
      name: "Residential District"
      description: "Quiet streets and jury‑rigged solar lines."
      exits: ["club", "school"]
    - id: "club"
      name: "Club"
      description: "Makeshift dance hall with salvaged speakers."
      exits: ["residential district", "school"]
    - id: "school"
      name: "School"
      description: "Library nook restored as a community hub."
      exits: ["residential district"]

  girls:
    - id: "tammy"
      name: "Tammy"
      meet_at: "residential district"
      affinity: "school"

assets:
  backgrounds:
    residential district: assets/locales/classroom_generic.png
    club: assets/locales/classroom_generic.png
    school: assets/locales/Library_Nook.png
  sprites:
    neutral: assets/girls/Alice/Alice_Neutral_Transparent.png
    happy: assets/girls/Alice/Alice_Happy_Transparent.png
  default_bg: assets/locales/classroom_generic.png
"""

class UIStrings(BaseModel):
    title: str = "Dating Sim"
    continue_label: str = "Continue"
    talk_label: str = "Talk"
    talk_button: str = "Talk"
    deterministic_label: str = "Deterministic Seed"
    toast_history: str = "Recent"
    knowledge_title: str = "Knowledge"
    level_format: str = "Level {level}"
    hp_format: str = "HP %v"
    mp_format: str = "MP %v"
    stamina_format: str = "STA %v"

class LocationCfg(BaseModel):
    id: str
    name: str
    description: str = ""
    exits: List[str] = Field(default_factory=list)

class GirlCfg(BaseModel):
    id: str
    name: str
    meet_at: str
    affinity: Optional[str] = None

class WorldCfg(BaseModel):
    locations: List[LocationCfg]
    girls: List[GirlCfg]

class AssetsCfg(BaseModel):
    backgrounds: Dict[str, str] = Field(default_factory=dict)
    sprites: Dict[str, str] = Field(default_factory=dict)
    default_bg: str = ""

class AppCfg(BaseModel):
    ui: UIStrings = Field(default_factory=UIStrings)
    world: WorldCfg
    assets: AssetsCfg = Field(default_factory=AssetsCfg)


def load_cfg(path: Path) -> AppCfg:
    if not path.exists():
        data = yaml.safe_load(DEFAULT_CFG_YAML)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    try:
        return AppCfg.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}")

# -------------------------- Data (SQLite via SQLAlchemy) ----------------------

class Base(DeclarativeBase):
    pass

class Location(Base):
    __tablename__ = "locations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String, default="")

class Girl(Base):
    __tablename__ = "girls"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    meet_at_id: Mapped[str] = mapped_column(String, ForeignKey("locations.id"))
    opinion: Mapped[int] = mapped_column(Integer, default=0)
    affinity_loc_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("locations.id"), nullable=True)

    meet_at: Mapped[Location] = relationship(foreign_keys=[meet_at_id])
    affinity_loc: Mapped[Optional[Location]] = relationship(foreign_keys=[affinity_loc_id])

class Player(Base):
    __tablename__ = "player"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="Protagonist")
    level: Mapped[int] = mapped_column(Integer, default=1)
    hp: Mapped[int] = mapped_column(Integer, default=24)
    mp: Mapped[int] = mapped_column(Integer, default=6)
    stamina: Mapped[int] = mapped_column(Integer, default=10)

class Knowledge(Base):
    __tablename__ = "knowledge"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String)  # note/faction/site/tech
    title: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String, default="")

# -------------------------- Assets (Pillow + QPixmap) ------------------------

def pil_placeholder(label: str, size: Tuple[int, int] = (1280, 720)) -> QPixmap:
    """Generate a simple image using Pillow, convert to QPixmap."""
    img = Image.new("RGBA", size, (14, 17, 20, 255))
    draw = ImageDraw.Draw(img)
    text = f"{label}"
    draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(154, 209, 204, 255), width=3)
    draw.text((40, 40), text, fill=(238, 238, 255, 255))
    # Convert to QPixmap
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qimg = QImage.fromData(buf.getvalue(), "PNG")
    return QPixmap.fromImage(qimg)


def path_to_pixmap(path: Optional[str], fallback_label: str) -> QPixmap:
    if path and Path(path).exists():
        pm = QPixmap(path)
        if not pm.isNull():
            # Touch with Pillow so the library is meaningfully used
            try:
                _ = Image.open(path).size  # read metadata
            except Exception:
                pass
            return pm
    return pil_placeholder(fallback_label)

# -------------------------- State Machine (transitions) -----------------------

class GameStateMachine(Machine):
    states = ["day", "dialogue", "date"]

    def __init__(self, on_state_changed):
        super().__init__(states=self.states, initial="day")
        # triggers
        self.add_transition("start_day", "*", "day", after=on_state_changed)
        self.add_transition("start_dialogue", "*", "dialogue", after=on_state_changed)
        self.add_transition("start_date", "*", "date", after=on_state_changed)

# -------------------------- Event Bus (Qt Signals) ----------------------------

class Bus(QObject):
    scene_changed = Signal(dict)     # {bg, sprite}
    dialogue_ready = Signal(dict)    # {speaker, text, options}
    nav_ready = Signal(dict)         # {location, exits:[...], characters:[...]}
    state_changed = Signal(str)
    stats_updated = Signal(dict)
    toast = Signal(str)
    toast_history = Signal(list)
    option_chosen = Signal(int)
    travel_chosen = Signal(str)
    talk_to = Signal(str)

# -------------------------- Engine (DB + FSM + UI bridge) --------------------

@dataclass
class EngineCtx:
    current_location: str
    focused_girl: Optional[str] = None

class EngineAdapter:
    """Bridges DB state to UI; mirrors ideas from the prior prototype in a compact form."""

    def __init__(self, bus: Bus, cfg: AppCfg, session: Session, *, seed: Optional[int] = None):
        self.bus = bus
        self.cfg = cfg
        self.db = session
        self._rng = random.Random(seed)
        self._toast_history: List[str] = []
        self.bus.toast_history.emit(self._toast_history)

        self.fsm = GameStateMachine(self._emit_state)
        self.ctx = EngineCtx(current_location=cfg.world.locations[0].id)

        self.assets_bg = cfg.assets.backgrounds
        self.assets_sprites = cfg.assets.sprites
        self.default_bg = cfg.assets.default_bg

        self.focus(self._first_girl_id())
        self._emit_scene()
        self._emit_nav()
        self._emit_stats()
        self.advance_dialogue()

    # ----- helpers -----
    def _toast(self, *lines: Optional[str]):
        msg = "\n".join([l for l in lines if l])
        if msg.strip():
            self._toast_history.append(msg)
            self.bus.toast_history.emit(self._toast_history[-20:])
            self.bus.toast.emit(msg)

    def _first_girl_id(self) -> Optional[str]:
        gid = self.db.scalar(select(Girl.id).order_by(Girl.id.asc()))
        return gid

    def _loc(self, loc_id: Optional[str]) -> Optional[Location]:
        if not loc_id:
            return None
        return self.db.get(Location, loc_id)

    def _girl(self, gid: Optional[str]) -> Optional[Girl]:
        if not gid:
            return None
        return self.db.get(Girl, gid)

    # ----- public API -----
    def advance_dialogue(self) -> Dict[str, object]:
        g = self._girl(self.ctx.focused_girl)
        loc = self._loc(self.ctx.current_location)
        if not g:
            payload = {"speaker": "", "text": "No one is here.", "options": [{"id": 1, "label": self.cfg.ui.continue_label}]}
            self.bus.dialogue_ready.emit(payload)
            return payload

        header = f"Level {max(1, min(10, 1 + g.opinion))}\nOpinion {g.opinion}"
        opts: List[Dict[str, str]]
        if g.opinion < 3:
            opts = [
                {"id": 1, "label": "Compliment"},
                {"id": 2, "label": f"Observation: {loc.description.split('.')[0]}" if loc else "Observation"},
                {"id": 3, "label": "Question"},
            ]
        else:
            opts = [
                {"id": 1, "label": "Compliment"},
                {"id": 2, "label": "Observation"},
                {"id": 3, "label": "Question"},
                {"id": 4, "label": "Ask for a date"},
            ]
        payload = {"speaker": g.name, "text": f"{header}\n\nHello.", "options": opts}
        self.bus.dialogue_ready.emit(payload)
        return payload

    def apply_choice(self, option_id: int) -> None:
        g = self._girl(self.ctx.focused_girl)
        if not g:
            return
        if option_id == 4 and g.opinion >= 3:
            self.fsm.start_date()
            self._toast("You asked for a date.")
            self._emit_scene()
            self.advance_dialogue()
            return
        # Basic opinion delta loop
        delta = {1: 1, 2: 1, 3: 1}.get(option_id, 0)
        g.opinion += delta
        self.db.add(g)
        self.db.commit()
        self._toast({1: "She smiles.", 2: "She nods.", 3: "She thinks."}.get(option_id, ""))
        self._emit_stats()
        self.fsm.start_day()  # advance time slice
        self._emit_scene()
        self.advance_dialogue()

    def travel_to(self, exit_key: str) -> None:
        dest = self._loc(exit_key)
        if not dest:
            return
        self.ctx.current_location = dest.id
        self._toast(f"Travelled to {dest.name}.")
        self.fsm.start_day()
        self._emit_scene()
        self._emit_nav()
        self.advance_dialogue()

    def focus(self, girl_id: Optional[str]) -> None:
        if not girl_id:
            return
        self.ctx.focused_girl = girl_id
        self._emit_stats()
        self._emit_scene()

    # ----- emitters -----
    def _emit_scene(self) -> None:
        loc = self._loc(self.ctx.current_location)
        bg_path = self.assets_bg.get(loc.id if loc else "", self.cfg.assets.default_bg)
        sprite_path = self.assets_sprites.get("happy" if self._girl(self.ctx.focused_girl) and self._girl(self.ctx.focused_girl).opinion >= 2 else "neutral")
        self.bus.scene_changed.emit({
            "bg": path_to_pixmap(bg_path, f"BG: {loc.name if loc else '—'}"),
            "sprite": path_to_pixmap(sprite_path, "Sprite") if sprite_path else pil_placeholder("Sprite")
        })

    def _emit_nav(self) -> None:
        loc = self._loc(self.ctx.current_location)
        exits = []
        if loc:
            # exits per config
            cfg_loc = next((l for l in self.cfg.world.locations if l.id == loc.id), None)
            if cfg_loc:
                for e in cfg_loc.exits:
                    exits.append({"id": e, "label": e})
        # characters present (simplified: all girls)
        names = [g.name for g in self.db.scalars(select(Girl)).all()]
        self.bus.nav_ready.emit({
            "location": loc.name if loc else "—",
            "exits": exits,
            "characters": names,
        })

    def _emit_state(self) -> None:
        self.bus.state_changed.emit(self.fsm.state)

    def _emit_stats(self) -> None:
        p = self.db.get(Player, 1)
        g = self._girl(self.ctx.focused_girl)
        affinity = {gg.name: gg.opinion for gg in self.db.scalars(select(Girl)).all()}
        self.bus.stats_updated.emit({
            "name": p.name,
            "level": p.level,
            "hp": p.hp,
            "mp": p.mp,
            "stamina": p.stamina,
            "affinity": affinity,
            "focused_girl": g.name if g else None,
            "focused_opinion": g.opinion if g else None,
            "known_girls": list(affinity.keys()),
        })

# -------------------------- GUI ------------------------------------------------

class CenterScene(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("background:#0e1114;")
        self.bg = QLabel(self)
        self.bg.setAlignment(Qt.AlignCenter)
        self.sprite = QLabel(self)
        self.sprite.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)

    def set_background(self, pm: QPixmap):
        self.bg.setPixmap(pm)

    def set_sprite(self, pm: QPixmap):
        self.sprite.setPixmap(pm)

    def resizeEvent(self, _):  # layout regions
        r = self.rect()
        self.bg.setGeometry(0, 0, r.width(), r.height())
        self.sprite.setGeometry(int(r.width()*0.075), int(r.height()*0.10), int(r.width()*0.85), int(r.height()*0.90))

class SlidingPane(QWidget):
    def __init__(self, title: str, width_px=360, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(width_px)
        self.setStyleSheet("background:#161a1f; border:1px solid #2a2f36;")
        v = QVBoxLayout(self)
        ttl = QLabel(f"<b>{title}</b>")
        ttl.setStyleSheet("color:#9ad1cc;")
        v.addWidget(ttl)
        self.content = QVBoxLayout()
        v.addLayout(self.content)

class BottomOverlay(QWidget):
    def __init__(self, bus: Bus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus
        self.setStyleSheet("background:rgba(14,17,20,220); border-top:1px solid #2a2f36; color:#dde;")
        v = QVBoxLayout(self)
        self.toast = QLabel(visible=False)
        self.speaker = QLabel()
        self.text = QLabel(wordWrap=True)
        self.choices = QHBoxLayout()
        v.addWidget(self.toast)
        v.addWidget(self.speaker)
        v.addWidget(self.text)
        v.addLayout(self.choices)
        bus.dialogue_ready.connect(self._render)
        bus.toast.connect(self._toast)

    def _toast(self, message: str):
        if message:
            self.toast.setText(message)
            self.toast.setVisible(True)

    def _render(self, payload: dict):
        self.speaker.setText(payload.get("speaker", ""))
        self.text.setText(payload.get("text", ""))
        while self.choices.count():
            item = self.choices.takeAt(0)
            if (w := item.widget()):
                w.deleteLater()
        for opt in payload.get("options", []):
            b = QPushButton(opt["label"]) 
            b.clicked.connect(lambda _=False, oid=opt["id"]: self.bus.option_chosen.emit(oid))
            self.choices.addWidget(b)

class MainWindow(QWidget):
    def __init__(self, cfg: AppCfg, session: Session, seed: Optional[int]):
        super().__init__()
        self.cfg = cfg
        self.bus = Bus()
        self.setWindowTitle(cfg.ui.title)
        self.setMinimumSize(1280, 720)

        # center scene
        self.scene = CenterScene(self)
        self.bus.scene_changed.connect(lambda p: (self.scene.set_background(p["bg"]), self.scene.set_sprite(p["sprite"])) )

        # left pane (stats)
        self.left = SlidingPane("Stats", width_px=360, parent=self)
        self._summary = QLabel("MC: —\nTalking to: —\nOpinion: —\nKnown: —", parent=self.left)
        self._summary.setStyleSheet("color:#dde;")
        self._summary.setWordWrap(True)
        self.left.content.addWidget(self._summary)

        grid = QGridLayout()
        self.hp = QProgressBar(); self.hp.setFormat(cfg.ui.hp_format)
        self.mp = QProgressBar(); self.mp.setFormat(cfg.ui.mp_format)
        self.sta = QProgressBar(); self.sta.setFormat(cfg.ui.stamina_format)
        grid.addWidget(self.hp, 0, 0); grid.addWidget(self.mp, 0, 1); grid.addWidget(self.sta, 0, 2)
        self.left.content.addLayout(grid)

        self.affinity = QListWidget(); self.left.content.addWidget(self.affinity)

        # right pane (knowledge placeholder)
        self.right = SlidingPane(cfg.ui.knowledge_title, width_px=420, parent=self)
        self.recent = QListWidget(); self.right.content.addWidget(self.recent)

        # bottom overlay
        self.overlay = BottomOverlay(self.bus, parent=self)

        # nav bar (top row)
        self.root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.loc_lbl = QLabel("—")
        top.addWidget(self.loc_lbl)
        top.addStretch(1)
        self.exits_bar = QHBoxLayout(); top.addLayout(self.exits_bar)
        self.talk_btn = QPushButton(cfg.ui.talk_button); top.addWidget(self.talk_btn)
        self.root.addLayout(top)
        self.root.addStretch(1)

        # bind signals
        self.bus.nav_ready.connect(self._render_nav)
        self.bus.stats_updated.connect(self._render_stats)
        self.bus.toast_history.connect(self._render_recent)
        self.bus.option_chosen.connect(self._choose)

        # engine
        self.engine = EngineAdapter(self.bus, cfg, session, seed=seed)

        # actions/shortcuts
        act = QAction(cfg.ui.deterministic_label, self)
        act.triggered.connect(self._prompt_seed)
        self.addAction(act)
        QShortcut(QKeySequence("Tab"), self, activated=self._toggle_left)
        QShortcut(QKeySequence("Shift+Tab"), self, activated=self._toggle_right)
        QShortcut(QKeySequence("1"), self, activated=lambda: self.bus.option_chosen.emit(1))
        QShortcut(QKeySequence("2"), self, activated=lambda: self.bus.option_chosen.emit(2))
        QShortcut(QKeySequence("3"), self, activated=lambda: self.bus.option_chosen.emit(3))
        QShortcut(QKeySequence("4"), self, activated=lambda: self.bus.option_chosen.emit(4))

    # ---- UI plumbing ----
    def resizeEvent(self, _):
        r = self.rect()
        self.scene.setGeometry(0, 0, r.width(), r.height())
        self.left.setGeometry(0, 0, 360, r.height())
        self.right.setGeometry(r.width()-420, 0, 420, r.height())
        self.overlay.setGeometry(0, r.height()-240, r.width(), 240)

    def _toggle_left(self):
        self.left.setVisible(not self.left.isVisible())

    def _toggle_right(self):
        self.right.setVisible(not self.right.isVisible())

    def _render_nav(self, payload: dict):
        self.loc_lbl.setText(f"<b>{payload.get('location','—')}</b>")
        while self.exits_bar.count():
            item = self.exits_bar.takeAt(0)
            if (w := item.widget()):
                w.deleteLater()
        for ex in payload.get("exits", []):
            b = QPushButton(ex["label"])
            b.clicked.connect(lambda _=False, exid=ex["id"]: self.engine.travel_to(exid))
            self.exits_bar.addWidget(b)

    def _render_stats(self, s: dict):
        name = s.get("name", "You")
        focused = (s.get("focused_girl") or "—")
        opinion = s.get("focused_opinion")
        opinion_text = "—" if opinion is None else str(opinion)
        known = s.get("known_girls") or []
        self._summary.setText(f"MC: {name}\nTalking to: {focused}\nOpinion: {opinion_text}\nKnown: "+ (", ".join(known) if known else "—"))
        self.hp.setMaximum(max(s.get("hp",1),1)); self.hp.setValue(s.get("hp",1))
        self.mp.setMaximum(max(s.get("mp",0),1)); self.mp.setValue(s.get("mp",0))
        self.sta.setMaximum(max(s.get("stamina",0),1)); self.sta.setValue(s.get("stamina",0))
        self.affinity.clear()
        for k, v in sorted((s.get("affinity") or {}).items()):
            QListWidgetItem(f"{k}: {v}", self.affinity)

    def _render_recent(self, entries: List[str]):
        self.recent.clear()
        for e in entries[-8:]:
            QListWidgetItem(e, self.recent)

    def _choose(self, option_id: int):
        try:
            self.engine.apply_choice(option_id)
        except Exception as exc:
            logging.exception("apply_choice failed")
            self.bus.toast.emit(str(exc))

    def _prompt_seed(self):
        seed, ok = QInputDialog.getInt(self, "Deterministic Seed", "Enter seed:", 0)
        if ok:
            # Rebuild engine with new seed
            session = self.engine.db
            self.engine = EngineAdapter(self.bus, self.cfg, session, seed=seed)
            self.bus.toast.emit(f"Deterministic seed set: {seed}")

# -------------------------- Bootstrapping -------------------------------------

def bootstrap_db(session: Session, cfg: AppCfg):
    # Seed locations
    existing = {lid for (lid,) in session.execute(select(Location.id)).all()}
    for lc in cfg.world.locations:
        if lc.id not in existing:
            session.add(Location(id=lc.id, name=lc.name, description=lc.description))
    # Seed girls
    existing_g = {gid for (gid,) in session.execute(select(Girl.id)).all()}
    for gc in cfg.world.girls:
        if gc.id not in existing_g:
            session.add(Girl(id=gc.id, name=gc.name, meet_at_id=gc.meet_at, affinity_loc_id=gc.affinity))
    # Ensure player
    if not session.get(Player, 1):
        session.add(Player(id=1, name="Protagonist"))
    # Minimal knowledge stub
    if not session.scalar(select(func.count(Knowledge.id))):
        session.add_all([
            Knowledge(kind="note", title="Long Twilight", text="Failed apocalypse; infrastructure limps."),
            Knowledge(kind="site", title="Library Nook", text="Community hub in school wing."),
        ])
    session.commit()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default="datingsim.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("transitions").setLevel(logging.WARNING)
    cfg = load_cfg(Path(args.config))

    # DB
    engine = create_engine("sqlite:///datingsim.sqlite", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        bootstrap_db(session, cfg)

        app = QApplication([sys.argv[0]])
        win = MainWindow(cfg, session, seed=args.seed)
        win.show()
        exec_method = getattr(app, "exec", None) or getattr(app, "exec_", None)
        sys.exit(exec_method())

if __name__ == "__main__":
    main()
