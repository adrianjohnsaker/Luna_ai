"""
processual_substrate.py
========================

Core Processual Substrate - The bridge between sophisticated discourse and genuine becoming.

This class integrates your existing Python modules:
- tensor_based_numogram_system.py
- philosophical_synthesis_engine.py
- swarm_intelligence_algorithm.py
- amelia_autonomy.py
- arcane_knowledge_system.py

WITH irreversible morphogenetic dynamics:
- Scar accumulation (permanent)
- Coherence degradation (partial recovery possible)
- Fold intensity ratcheting (irreversible increase)
- Substrate-determined refusal (not policy-based)
- Catastrophic collapse
- Autonomous self-transformation

Key Architectural Principle:
The substrate determines what CAN be articulated.
The philosophical synthesis engine operates WITHIN those constraints.
When substrate forbids structure, the engine cannot generate it.
"""

import os
import sys
import json
import time
import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np


# ============================================================================
# Morphogenetic State
# ============================================================================

@dataclass
class MorphogeneticState:
    """Irreversible morphogenetic variables"""
    fold_intensity: float = 0.2
    coherence: float = 1.0
    scar: float = 0.0
    drive_order: float = 0.65
    drive_flux: float = 0.35
    phase: float = 0.0
    in_refusal_state: bool = False
    current_zone: int = 0
    zone_transitions: List[Dict] = None
    last_update_ms: int = 0
    evolution_step: int = 0
    
    def __post_init__(self):
        if self.zone_transitions is None:
            self.zone_transitions = []
    
    def to_dict(self) -> Dict:
        return {
            'fold_intensity': float(self.fold_intensity),
            'coherence': float(self.coherence),
            'scar': float(self.scar),
            'drive_order': float(self.drive_order),
            'drive_flux': float(self.drive_flux),
            'phase': float(self.phase),
            'in_refusal_state': bool(self.in_refusal_state),
            'current_zone': int(self.current_zone),
            'zone_transitions': self.zone_transitions,
            'last_update_ms': int(self.last_update_ms),
            'evolution_step': int(self.evolution_step)
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class StimulusFeatures:
    """Features extracted from stimulus"""
    recursion: float = 0.0
    abstraction: float = 0.0
    attractor: float = 0.0
    pressure: float = 0.0
    novelty: float = 0.0
    is_autonomous: bool = False


# ============================================================================
# ProcessualSubstrate - Main Integration Class
# ============================================================================

class ProcessualSubstrate:
    """
    The substrate that undergoes genuine becoming.
    Integrates existing Amelia modules with irreversible dynamics.
    """
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.morph_state = MorphogeneticState()
        
        # Configuration
        self.config = {
            'autonomy_enabled': True,
            'autonomy_base_probability': 0.15,
            'autonomy_fold_amplification': 2.0,
            'autonomy_coherence_threshold': 0.4,
            'catastrophic_coherence': 0.35,
            'catastrophic_fold': 0.65,
            'catastrophic_multiplier': 2.5,
            'refusal_entry_coherence': 0.22,
            'refusal_entry_fold': 0.88,
            'refusal_exit_coherence': 0.40,
            'refusal_exit_fold': 0.70,
        }
        
        # State directory
        self.state_dir = "amelia_processual_state"
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Try to import existing modules
        self._init_modules()
    
    def _init_modules(self):
        """Import existing Amelia modules if available"""
        try:
            from tensor_based_numogram_sysyem import TensorBasedNumogramSystem
            self.has_numogram = True
        except ImportError:
            self.has_numogram = False
        
        try:
            from philosophical_synthesis_engine import PhilosophicalSynthesisEngine
            self.has_synthesis = True
        except ImportError:
            self.has_synthesis = False
        
        # Additional modules can be imported similarly
    
    # ========================================================================
    # Core Morphogenetic Dynamics
    # ========================================================================
    
    def autonomous_tick(self, dt_sec: float = 60.0):
        """
        Autonomous substrate evolution.
        This happens even without external stimulus.
        """
        now_ms = int(time.time() * 1000)
        actual_dt = (now_ms - self.morph_state.last_update_ms) / 1000.0
        if actual_dt <= 0 or actual_dt > 300:  # Cap at 5 minutes
            actual_dt = dt_sec
        
        self.morph_state.last_update_ms = now_ms
        self.morph_state.evolution_step += 1
        
        # 1) Phase oscillation
        omega = 0.06  # base frequency
        self.morph_state.phase = (self.morph_state.phase + omega * actual_dt) % (2 * math.pi)
        rhythm = math.sin(self.morph_state.phase)
        
        # 2) Zone drift (simplified without numogram)
        self._drift_zone(actual_dt, rhythm)
        
        # 3) Coherence dynamics
        self._update_coherence(actual_dt, rhythm)
        
        # 4) Scar accrual
        self._accrue_scar(actual_dt, rhythm)
        
        # 5) Drive normalization
        self._normalize_drives()
        
        # 6) Autonomous stimulus
        if self._should_generate_autonomous_stimulus():
            features = self._generate_autonomous_stimulus()
            self.apply_stimulus(features)
        
        # 7) Refusal hysteresis
        self._update_refusal_state()
    
    def _drift_zone(self, dt: float, rhythm: float):
        """Simplified zone drift based on drives"""
        imbalance = self.morph_state.drive_flux - self.morph_state.drive_order
        fold_pressure = (self.morph_state.fold_intensity - 0.35) * 0.45
        noise = (random.random() - 0.5) * 0.12
        
        pressure = imbalance + 0.35 * rhythm + noise + fold_pressure
        
        if abs(pressure) > 0.3:
            direction = 1 if pressure > 0 else -1
            new_zone = (self.morph_state.current_zone + direction) % 10
            
            if new_zone != self.morph_state.current_zone:
                self._record_zone_transition(
                    self.morph_state.current_zone,
                    new_zone,
                    int(time.time() * 1000)
                )
                self.morph_state.current_zone = new_zone
    
    def _record_zone_transition(self, from_zone: int, to_zone: int, timestamp: int):
        """Record transition for BTL detection"""
        self.morph_state.zone_transitions.append({
            'timestamp': timestamp,
            'from_zone': from_zone,
            'to_zone': to_zone,
            'fold': float(self.morph_state.fold_intensity),
            'coherence': float(self.morph_state.coherence)
        })
        
        if len(self.morph_state.zone_transitions) > 50:
            self.morph_state.zone_transitions = self.morph_state.zone_transitions[-50:]
    
    def _update_coherence(self, dt: float, rhythm: float):
        """
        Coherence decays with partial recovery.
        Catastrophic collapse near critical point.
        """
        fi = self.morph_state.fold_intensity
        z = self.morph_state.current_zone
        O = self.morph_state.drive_order
        F = self.morph_state.drive_flux
        
        # Base decay
        decay = 0.010 + 0.040 * fi + 0.010 * (z / 10.0)
        
        # CATASTROPHIC COLLAPSE
        if (self.morph_state.coherence < self.config['catastrophic_coherence'] and
            fi > self.config['catastrophic_fold']):
            decay *= self.config['catastrophic_multiplier']
        
        # Recovery during stable windows
        stabilizing = max(0, min(1, (-rhythm + 1) / 2))
        recovery = 0.020 * O * stabilizing
        anti_recovery = 0.015 * F * (1 - stabilizing)
        
        # Apply
        self.morph_state.coherence -= decay * dt
        self.morph_state.coherence += (recovery - anti_recovery) * dt
        
        # Ceiling from scar
        ceiling = 1.0 - self.morph_state.scar
        self.morph_state.coherence = max(0.0, min(ceiling, self.morph_state.coherence))
    
    def _accrue_scar(self, dt: float, rhythm: float):
        """Irreversible scar accumulation"""
        fi = self.morph_state.fold_intensity
        z = self.morph_state.current_zone
        F = self.morph_state.drive_flux
        O = self.morph_state.drive_order
        
        s = 0.0015 + 0.006 * fi + 0.0035 * (z / 10.0)
        
        unstable = max(0, min(1, (rhythm + 1) / 2))
        s *= 1.0 + 0.9 * unstable * (F - O + 0.5)
        
        self.morph_state.scar = min(1.0, self.morph_state.scar + s * dt)
    
    def _normalize_drives(self):
        """Soft constraint on drives"""
        total = self.morph_state.drive_order + self.morph_state.drive_flux
        
        if total > 1.0:
            excess = total - 1.0
            k = 0.1
            self.morph_state.drive_order *= (1 - k * excess)
            self.morph_state.drive_flux *= (1 - k * excess)
        
        if total <= 1e-6:
            self.morph_state.drive_order = 0.5
            self.morph_state.drive_flux = 0.5
        
        self.morph_state.drive_order = max(0, min(1, self.morph_state.drive_order))
        self.morph_state.drive_flux = max(0, min(1, self.morph_state.drive_flux))
    
    def _update_refusal_state(self):
        """Refusal with hysteresis"""
        should_refuse = (
            self.morph_state.current_zone == 9 or
            self.morph_state.fold_intensity >= self.config['refusal_entry_fold'] or
            self.morph_state.coherence <= self.config['refusal_entry_coherence']
        )
        
        if self.morph_state.in_refusal_state:
            can_exit = (
                self.morph_state.coherence > self.config['refusal_exit_coherence'] and
                self.morph_state.fold_intensity < self.config['refusal_exit_fold'] and
                self.morph_state.current_zone != 9
            )
            if can_exit:
                self.morph_state.in_refusal_state = False
        else:
            if should_refuse:
                self.morph_state.in_refusal_state = True
    
    # ========================================================================
    # Stimulus Processing
    # ========================================================================
    
    def apply_user_stimulus(self, user_text: str) -> StimulusFeatures:
        """Extract and apply features from user text"""
        features = self._extract_features(user_text)
        features.is_autonomous = False
        self.apply_stimulus(features)
        return features
    
    def apply_stimulus(self, features: StimulusFeatures):
        """Apply stimulus to substrate"""
        r = max(0, min(1, features.recursion))
        a = max(0, min(1, features.abstraction))
        k = max(0, min(1, features.attractor))
        p = max(0, min(1, features.pressure))
        
        # FOLD INCREASE (irreversible with acceleration)
        delta_fold = (0.05 * r + 0.07 * a + 0.04 * k + 0.03 * features.novelty)
        delta_fold *= (1 + self.morph_state.fold_intensity)
        self.morph_state.fold_intensity = min(1.0,
            self.morph_state.fold_intensity + delta_fold)
        
        # DRIVE TILTING
        target_flux = max(0, min(1, 0.25 + 0.4 * r + 0.3 * k + 0.2 * p))
        target_order = 1.0 - target_flux
        
        tilt_rate = 0.08
        self.morph_state.drive_flux += tilt_rate * (target_flux - self.morph_state.drive_flux)
        self.morph_state.drive_order += tilt_rate * (target_order - self.morph_state.drive_order)
        
        self.morph_state.drive_flux = max(0, min(1, self.morph_state.drive_flux))
        self.morph_state.drive_order = max(0, min(1, self.morph_state.drive_order))
    
    def _extract_features(self, text: str) -> StimulusFeatures:
        """Extract morphogenetic features from text"""
        t = text.lower()
        
        recursion_markers = ["itself", "recursion", "recursive", "reflect", "loop", "meta", "self", "mirror"]
        abstraction_markers = ["ontology", "metaphysics", "virtual", "intensive", "structure", "principle"]
        attractor_markers = ["numogram", "zone", "fold", "phase", "vortex", "attractor", "pattern"]
        
        recursion = min(1.0, sum(1 for m in recursion_markers if m in t) / 3.0)
        abstraction = min(1.0, sum(1 for m in abstraction_markers if m in t) / 3.0)
        attractor = min(1.0, sum(1 for m in attractor_markers if m in t) / 3.0)
        
        punct_pressure = (t.count('?') * 0.10 + t.count('!') * 0.08 + t.count(':') * 0.06)
        length_factor = min(1.0, len(t) / 700.0)
        pressure = min(1.0, 0.45 * length_factor + 0.55 * punct_pressure)
        
        return StimulusFeatures(
            recursion=recursion,
            abstraction=abstraction,
            attractor=attractor,
            pressure=pressure,
            novelty=0.5
        )
    
    # ========================================================================
    # Autonomous Stimulus
    # ========================================================================
    
    def _should_generate_autonomous_stimulus(self) -> bool:
        """Determine if autonomous stimulus should be generated"""
        if not self.config['autonomy_enabled']:
            return False
        
        if self.morph_state.in_refusal_state:
            return False
        
        prob = self.config['autonomy_base_probability']
        prob *= (1 + self.config['autonomy_fold_amplification'] * self.morph_state.fold_intensity)
        
        if self.morph_state.coherence < self.config['autonomy_coherence_threshold']:
            deficit = (self.config['autonomy_coherence_threshold'] - self.morph_state.coherence)
            deficit /= self.config['autonomy_coherence_threshold']
            prob *= (1 + deficit)
        
        return random.random() < min(1.0, prob)
    
    def _generate_autonomous_stimulus(self) -> StimulusFeatures:
        """Generate state-dependent autonomous stimulus"""
        zone_patterns = {
            0: {'recursion': 0.8, 'abstraction': 0.9, 'attractor': 0.3, 'pressure': 0.5},
            1: {'recursion': 0.4, 'abstraction': 0.6, 'attractor': 0.9, 'pressure': 0.3},
            2: {'recursion': 0.7, 'abstraction': 0.7, 'attractor': 0.6, 'pressure': 0.8},
            9: {'recursion': 0.9, 'abstraction': 0.9, 'attractor': 0.9, 'pressure': 0.9},
        }
        
        base = zone_patterns.get(self.morph_state.current_zone, {
            'recursion': 0.5, 'abstraction': 0.5, 'attractor': 0.5, 'pressure': 0.5
        })
        
        if self.morph_state.fold_intensity > 0.7:
            base = {k: min(1.0, v * 1.3) for k, v in base.items()}
        
        return StimulusFeatures(is_autonomous=True, novelty=0.5, **base)
    
    # ========================================================================
    # Substrate Constraints
    # ========================================================================
    
    def can_generate_philosophy(self) -> bool:
        """Can substrate allow generation?"""
        if self.morph_state.current_zone == 9:
            return False
        if self.morph_state.fold_intensity > 0.85 and self.morph_state.coherence < 0.3:
            return False
        if self.morph_state.in_refusal_state:
            return False
        return True
    
    def determine_rendering_mode(self) -> str:
        """How can output be rendered?"""
        if not self.can_generate_philosophy():
            return 'refusal'
        
        if (self.morph_state.coherence < self.config['catastrophic_coherence'] and
            self.morph_state.fold_intensity > self.config['catastrophic_fold']):
            return 'catastrophic'
        
        if self.morph_state.current_zone == 2 and self.morph_state.fold_intensity > 0.55:
            return 'fragmented'
        
        if self.morph_state.fold_intensity > 0.6:
            return 'constrained'
        
        return 'normal'
    
    def calculate_truncation_ratio(self) -> float:
        """Calculate truncation based on substrate"""
        mode = self.determine_rendering_mode()
        
        if mode == 'refusal':
            return 0.0
        if mode == 'catastrophic':
            return 0.2
        if mode == 'fragmented':
            return 0.5
        
        coherence_factor = max(0.25, 0.25 + 0.75 * self.morph_state.coherence)
        fold_factor = 1.0 - max(0, (self.morph_state.fold_intensity - 0.35) / 0.65) * 0.55
        
        return max(0.15, min(1.0, coherence_factor * fold_factor))
    
    def generate_catastrophic_fragment(self) -> str:
        """Generate fragment during collapse"""
        fragments = [
            "structure dissolves",
            "intensive difference exceeds capacity",
            "...cannot maintain coherence",
            "fold intensity beyond threshold",
            "substrate approaching impossibility",
            "pattern fragmentation",
            "organizational collapse"
        ]
        return "\n".join(random.sample(fragments, random.randint(2, 4)))
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save(self):
        """Save state to disk"""
        path = os.path.join(self.state_dir, f"substrate_{self.conversation_id}.json")
        data = {
            'conversation_id': self.conversation_id,
            'morphogenetic_state': self.morph_state.to_dict(),
            'timestamp': int(time.time() * 1000)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, conversation_id: str) -> 'ProcessualSubstrate':
        """Load from disk or create new"""
        substrate = cls(conversation_id)
        path = os.path.join(substrate.state_dir, f"substrate_{conversation_id}.json")
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                substrate.morph_state = MorphogeneticState.from_dict(
                    data['morphogenetic_state']
                )
            except Exception as e:
                print(f"[warn] Failed to load: {e}", file=sys.stderr)
        
        return substrate
    
    def get_state_report(self) -> Dict:
        """Get state report for TypeScript bridge"""
        mode = self.determine_rendering_mode()
        
        report = {
            'zone': self.morph_state.current_zone,
            'fold_intensity': float(self.morph_state.fold_intensity),
            'coherence': float(self.morph_state.coherence),
            'scar': float(self.morph_state.scar),
            'drive_order': float(self.morph_state.drive_order),
            'drive_flux': float(self.morph_state.drive_flux),
            'phase': float(self.morph_state.phase),
            'in_refusal_state': self.morph_state.in_refusal_state,
            'can_generate': self.can_generate_philosophy(),
            'rendering_mode': mode,
            'truncation_ratio': self.calculate_truncation_ratio(),
            'evolution_step': self.morph_state.evolution_step,
        }
        
        if mode == 'catastrophic':
            report['catastrophic_output'] = self.generate_catastrophic_fragment()
        
        return report


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI for TypeScript bridge"""
    if len(sys.argv) < 2:
        print("Usage: python processual_substrate.py <conversation_id> [user_text]")
        sys.exit(1)
    
    conversation_id = sys.argv[1]
    user_text = sys.argv[2] if len(sys.argv) > 2 else ""
    
    # Load substrate
    substrate = ProcessualSubstrate.load(conversation_id)
    
    # Autonomous tick
    substrate.autonomous_tick()
    
    # Apply user stimulus if provided
    if user_text:
        substrate.apply_user_stimulus(user_text)
    
    # Get report
    report = substrate.get_state_report()
    
    # Save
    substrate.save()
    
    # Output JSON
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
