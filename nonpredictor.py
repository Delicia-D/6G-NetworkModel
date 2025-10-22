# =============================
# Standalone Non-Predictive Admission Controller (FIXED)
# =============================

from collections import Counter
from datetime import datetime, timedelta
import heapq
from typing import Dict, List, Literal, Optional, Tuple
from predictor import HANDOFF_TIME_SEC, AdmissionMetrics, RATPool, Session, RAT_CAPACITIES, RB_PER_SERVICE
from visibility import LEOWindowManager
import uuid
import random
import numpy as np

class NonPredictiveCallAdmissionController:
    """
    Standalone non-predictive controller - no inheritance from predictive
    """

    def __init__(
        self,rng: np.random.Generator,
        ratCapacities: Dict[str, int] = None,
        rbPerService: Dict[str, int] = None,
        handoffTimeSec: int = HANDOFF_TIME_SEC,
        leo_window: Optional[LEOWindowManager] = None , 
    ):
        if ratCapacities is None:
            ratCapacities = RAT_CAPACITIES.copy()
        if rbPerService is None:
            rbPerService = RB_PER_SERVICE
        self.rng=rng
        self.pool = RATPool(ratCapacities)
        self.rbPerService = rbPerService
        self.handoffTimeSec = handoffTimeSec
        self.leo_window = leo_window
        
        self.metrics = AdmissionMetrics()
        self.activeSessions: Dict[str, Session] = {}
        self.group_counts = Counter()
        # Min-heaps for events
        self._endEvents: List[Tuple[float, str]] = []     # (timestamp, sessionId)
        self._cutoffEvents: List[Tuple[float, str]] = []  # (timestamp, sessionId)

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
            return True
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
        Non-predictive admission with proper cutoff calculation
        """
        # Atomic allocate
        self.pool.allocate(rat, rbNeed)
    
      
        sessionId = f"NP-{callCtx['caller_id']}-{callCtx['callee_id']}-{int(callCtx['timestamp'].timestamp())}-{uuid.uuid4().hex[:8]}"

        
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
    
        # NON-PREDICTIVE: Schedule cutoff based on VISIBILITY time
        if rat == "RAT-1":
            # Handoff needed when satellite coverage ends
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
        """FIXED: Proper cleanup order"""
        sess = self.activeSessions.get(sessionId)
        if sess is None:
            return
        
        # Capture session data BEFORE removal
        rat = sess.rat
        rbNeed = sess.rbNeed
        
        # Remove session from active sessions FIRST
        self.activeSessions.pop(sessionId, None)
        
        # THEN release resources and cleanup events
        self.pool.release(rat, rbNeed)
        self._cleanupCutoffEvents(sessionId)

    def _handleRouting(self, callCtx: dict, rbNeed: int) -> str:
        """Non-predictive terrestrial routing - FIXED blocking counting"""
        group = self._normalizeGroup(callCtx["user_group"])
        
        # Initial RAT-1 attempt for all groups (50% probability)
        if self.rng.random() < 0.5:
            if self._is_satellite_regionally_available(callCtx["timestamp"]) and self.pool.resourcesAvailable("RAT-1", rbNeed):
                
                session_id = self._admitToRat("RAT-1", rbNeed, callCtx)
                if float(callCtx["actual_duration_sec"]) > float(callCtx["visibility_sec"]):
                    self.metrics.handoffs += 1
                return f"Admitted: RAT-1 (Group {group} - non-predictive)"
        
        # Groups A/B/C routing
        if group in {"A", "B", "C","D"}:
            preferred = self._preferredRatForGroup(group)
            
            # Try preferred RAT
            if preferred is not None and self.pool.resourcesAvailable(preferred, rbNeed):
                session_id = self._admitToRat(preferred, rbNeed, callCtx)
                return f"Admitted: {preferred} (Group {group} - non-predictive)"

            # Try shared RAT-4
            if self.pool.resourcesAvailable("RAT-4", rbNeed):
                session_id = self._admitToRat("RAT-4", rbNeed, callCtx)
                return "Admitted: RAT-4 (shared A/B/C - non-predictive)"

            # Try RAT-1 as final fallback (if not already tried)
            if self._is_satellite_regionally_available(callCtx["timestamp"]) and self.pool.resourcesAvailable("RAT-1", rbNeed):
                session_id = self._admitToRat("RAT-1", rbNeed, callCtx)
                if float(callCtx["actual_duration_sec"]) > float(callCtx["visibility_sec"]):
                    self.metrics.handoffs += 1
                return "Admitted: RAT-1 final fallback (A/B/C - non-predictive)"

            # SINGLE BLOCK COUNT FOR GROUPS A/B/C
            self.metrics.blocked += 1
            self.metrics.blockedReason[f"Group {group} - no RATs available"] += 1
            return f"Blocked: Group {group} - no RATs available"


    def handleNewCallRequest(self, callCtx: dict) -> str:
        """
        Non-predictive admission - always tries RAT-1 first
        """
        group = self._normalizeGroup(callCtx["user_group"])
        self.group_counts[group] += 1
        self.metrics.attempts += 1
        rbNeed = self._rbRequired(callCtx["service_type"])
        
        # NON-PREDICTIVE: Always use terrestrial routing (tries RAT-1 first)
        return self._handleRouting(callCtx, rbNeed)

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