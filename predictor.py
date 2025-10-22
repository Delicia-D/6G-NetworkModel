from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import heapq
import uuid
from visibility import LEOWindowManager

# =============================
# Constants (updated)
# =============================
#nsn
RAT_CAPACITIES: Dict[str, int] = {
    "RAT-1": 160,  # LEO satellite
    "RAT-2": 50,
    "RAT-3": 75,
    "RAT-4": 106,
    "RAT-5": 51,
}

RB_PER_SERVICE: Dict[str, int] = {
    "voice": 1,
    "video": 8,
}

HANDOFF_TIME_SEC: int = 30  # Time needed to complete handoff process

# =============================
# Data structures (updated - removed plannedHandoff)
# =============================

@dataclass
class RATState:
    capacity: int
    inUse: int = 0

@dataclass
class Session:
    sessionId: str
    rat: str
    rbNeed: int
    callerId: int
    calleeId: int
    serviceType: str
    startTime: datetime
    endTime: datetime
    userGroup: Literal["A","B","C","D"]  # Removed "D"
    preferredRAT: Optional[str] = None  # A->RAT-5, B->RAT-2, C->RAT-4
    sharedRAT: str = "RAT-4"            # shared for A/B/C/E

@dataclass
class AdmissionMetrics:
    def __init__(self):
        self.attempts: int = 0
        self.admitted: int = 0
        self.blocked: int = 0
      
        self.calls_admitted_to_satellite: int = 0
        # NEW HANDOFF METRICS
        self.handoffs: int = 0  # total attempts (successful + failed)

        # REMOVE field() - just use regular attributes
        self.admittedByRAT: Counter = Counter()
        self.blockedReason: Counter = Counter()
        # NEW: Track voice and video call admissions
        self.voice_calls_admitted: int = 0
        self.video_calls_admitted: int = 0
       
   

# =============================
# RAT Pool (unchanged)
# =============================

class RATPool:
    def __init__(self, capacities: Dict[str, int]):
        self.state: Dict[str, RATState] = {rat: RATState(cap) for rat, cap in capacities.items()}

    def resourcesAvailable(self, rat: str, rbNeed: int) -> bool:
        state = self.state[rat]
        available = state.capacity - state.inUse
        return available >= rbNeed

    def allocate(self, rat: str, rbNeed: int) -> None:
        state = self.state[rat]
        if state.inUse + rbNeed > state.capacity:
            raise RuntimeError(f"Over-allocation on {rat}: need {rbNeed}, have {state.capacity - state.inUse}")
        state.inUse = state.inUse + rbNeed

    def release(self, rat: str, rbNeed: int) -> None:
        state = self.state[rat]
        newInUse = state.inUse - rbNeed
        if newInUse < 0:
            raise RuntimeError(f"Negative inUse on {rat} after release: {newInUse}")
        state.inUse = newInUse

    def snapshot(self) -> Dict[str, Tuple[int, int]]:
        return {rat: (st.inUse, st.capacity) for rat, st in self.state.items()}


# =============================
# Admission Controller (FIXED)
# =============================

class PredictiveCallAdmissionController:
    """
    Handles admit/block decisions, resource accounting, and events:
    - End events: release RBs at call completion
    - LEO cutoff events: attempt handoff before visibility ends
    """

    def __init__(
        self,
        ratCapacities: Dict[str, int] = None,
        rbPerService: Dict[str, int] = None,
        handoffTimeSec: int = HANDOFF_TIME_SEC,
        leo_window: Optional[LEOWindowManager] = None 
    ):
        if ratCapacities is None:
            ratCapacities = RAT_CAPACITIES
        if rbPerService is None:
            rbPerService = RB_PER_SERVICE

        self.pool = RATPool(ratCapacities)
        self.rbPerService = rbPerService
        self.handoffTimeSec = handoffTimeSec  # Time needed for handoff process
        self.leo_window = leo_window 
        
        self.metrics = AdmissionMetrics()
        self.activeSessions: Dict[str, Session] = {}

        # Min-heaps for events (time-ordered)
        self._endEvents: List[Tuple[float, str]] = []     # (timestamp float, sessionId)
        self._cutoffEvents: List[Tuple[float, str]] = []  # (timestamp float, sessionId)
        self.group_counts = Counter()

    # ---------- Helpers ----------

    @staticmethod
    def _normalizeGroup(group: str) -> Literal["A","B","C","D"]:  # Removed "D"
       
        g = str(group).strip().upper()
        if g.startswith("GROUP "):
            g = g.split(" ", 1)[1]
        return g
        
    def _rbRequired(self, serviceType: str) -> int:
        if serviceType not in self.rbPerService:
            raise ValueError(f"Unknown service type: {serviceType}")
        return int(self.rbPerService[serviceType])

    def _is_satellite_regionally_available(self, timestamp: datetime) -> bool:
        """Check if satellite is available in the region at given time"""
        if self.leo_window is None:
            return True  # Backward compatibility
        return self.leo_window.is_available(timestamp.timestamp())

    def _preferredRatForGroup(self, group: str) -> Optional[str]:
        if group == "A":
            return "RAT-5"
        if group == "B":
            return "RAT-2"
        if group == "C":
            return "RAT-4"
        if group =="D":
            return "RAT-3"
        
        return None

    def _cleanupCutoffEvents(self, sessionId: str):
        """Remove a session's cutoff events from the heap"""
        self._cutoffEvents = [(t, sid) for t, sid in self._cutoffEvents if sid != sessionId]
        heapq.heapify(self._cutoffEvents)

    def _admitToRat(self, rat: str, rbNeed: int, callCtx: dict) -> str:
        """
        Allocate RBs and create a Session.
        Uses handoff time margin ONLY for cutoff scheduling, not admission decisions.
        """
        # Atomic allocate
        self.pool.allocate(rat, rbNeed)
    
        
        sessionId = f"{callCtx['caller_id']}-{callCtx['callee_id']}-{int(callCtx['timestamp'].timestamp())}-{uuid.uuid4().hex[:8]}"

        group = self._normalizeGroup(callCtx["user_group"])
        preferred = self._preferredRatForGroup(group)
    
        # Use ACTUAL duration for resource holding
        endTime = callCtx["timestamp"] + timedelta(seconds=float(callCtx["actual_duration_sec"]))
    
        sess = Session(
            sessionId=sessionId,
            rat=rat,
            rbNeed=rbNeed,
            callerId=int(callCtx["caller_id"]),
            calleeId=int(callCtx["callee_id"]),
            serviceType=str(callCtx["service_type"]),
            startTime=callCtx["timestamp"],
            endTime=endTime,
            userGroup=group,
            preferredRAT=preferred,
            sharedRAT="RAT-4",
        )
        self.activeSessions[sessionId] = sess
    
        # Schedule normal end release
        heapq.heappush(self._endEvents, (sess.endTime.timestamp(), sessionId))
    
        # Schedule LEO cutoff with handoff time margin
        if rat == "RAT-1":
            # Schedule handoff attempt BEFORE visibility ends
            tCut = callCtx["timestamp"] + timedelta(
                seconds=max(0.0, float(callCtx["visibility_sec"]) - self.handoffTimeSec)
            )
            heapq.heappush(self._cutoffEvents, (tCut.timestamp(), sessionId))
            self.metrics.calls_admitted_to_satellite += 1
        
        # Metrics
        self.metrics.admitted += 1
        self.metrics.admittedByRAT[rat] += 1

        # NEW: Track voice vs video admissions
        service_type = callCtx["service_type"].lower()
        if service_type == "voice":
            self.metrics.voice_calls_admitted += 1
        elif service_type == "video":
            self.metrics.video_calls_admitted += 1
            
        return sessionId

    def _releaseFromRat(self, sessionId: str) -> None:
        sess = self.activeSessions.get(sessionId)
        if sess is None:
            return
        self.pool.release(sess.rat, sess.rbNeed)
        # Clean up cutoff events
        self._cleanupCutoffEvents(sessionId)
        # Remove session
        self.activeSessions.pop(sessionId, None)

    def _handleTerrestrialRouting(self, callCtx: dict, rbNeed: int) -> str:
        """Fixed terrestrial routing with consistent blocking counting"""
        group = self._normalizeGroup(callCtx["user_group"])
        # Groups A/B/C routing
        if group in {"A", "B", "C", "D"}:
            preferred = self._preferredRatForGroup(group)
            
            # Try preferred RAT
            if preferred is not None and self.pool.resourcesAvailable(preferred, rbNeed):
                session_id = self._admitToRat(preferred, rbNeed, callCtx)
                return f"Admitted: {preferred} (Group {group})"

            # Try shared RAT-4
            if self.pool.resourcesAvailable("RAT-4", rbNeed):
                session_id = self._admitToRat("RAT-4", rbNeed, callCtx)
                return "Admitted: RAT-4 (shared A/B/C/D)"

            # Try RAT-1 as final fallback
            if self._is_satellite_regionally_available(callCtx["timestamp"]) and self.pool.resourcesAvailable("RAT-1", rbNeed) and (float(callCtx["predicted_duration_sec"]) > float(callCtx["visibility_sec"])):
                session_id = self._admitToRat("RAT-1", rbNeed, callCtx)
                #if float(callCtx["actual_duration_sec"]) > float(callCtx["visibility_sec"]):
                self.metrics.handoffs += 1
                return "Admitted: RAT-1 fallback (A/B/C/D)"

            # SINGLE BLOCK COUNT FOR GROUPS A/B/C
            self.metrics.blocked += 1
            self.metrics.blockedReason[f"Group {group} - terrestrial unavailable"] += 1
            return f"Blocked: Group {group} - terrestrial unavailable"

        """# This should never be reached due to group normalization
        self.metrics.blocked += 1
        self.metrics.blockedReason["Unknown group"] += 1
        return "Blocked: Unknown user group"""
    def handleNewCallRequest(self, callCtx: dict) -> str:
        """
        callCtx must include:
          - caller_id, callee_id, service_type ("voice"/"video"),
          - predicted_duration_sec (float),
          - visibility_sec (float, for RAT-1 at caller location & current pass),
          - user_group ("A".."E" or "Group A"..),
          - timestamp (datetime)
        """
        group = self._normalizeGroup(callCtx["user_group"])

        self.group_counts[group] += 1
        self.metrics.attempts += 1
        rbNeed = self._rbRequired(callCtx["service_type"])
       
        # FIXED:
        if self._is_satellite_regionally_available(callCtx["timestamp"]) and float(callCtx["predicted_duration_sec"]) <= float(callCtx["visibility_sec"]):
            if self.pool.resourcesAvailable("RAT-1", rbNeed):
                session_id = self._admitToRat("RAT-1", rbNeed, callCtx)
                return "Admitted: RAT-1 (short call)"
            else:
                return self._handleTerrestrialRouting(callCtx, rbNeed)
        
              # Track handoff need
        return self._handleTerrestrialRouting(callCtx, rbNeed)
    def releaseDueSessions(self, now: datetime):
        """Process end events only """
        nowTs = now.timestamp()
        
        sessions_to_release = []
        while self._endEvents and self._endEvents[0][0] <= nowTs:
            _, sessionId = heapq.heappop(self._endEvents)
            sessions_to_release.append(sessionId)
        
        for sessionId in sessions_to_release:
            if sessionId in self.activeSessions:
                self._releaseFromRat(sessionId)
       
    def getMetrics(self) -> AdmissionMetrics:
        return self.metrics

    def getRATSnapshot(self) -> Dict[str, Tuple[int, int]]:
        return self.pool.snapshot()