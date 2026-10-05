"""
Analytics engine for WarehouseVision AI.

Aggregates detection results, occupancy, zone violations,
tracking data, and inventory into a structured analytics payload.
Generates data-driven insights based on actual analysis results.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class WarehouseAnalytics:
    """Aggregates and computes warehouse analytics from detection results."""

    def compute(
        self,
        object_counts: Dict[str, int],
        occupancy_data: Dict[str, Any],
        zone_violations: Dict[str, Any],
        inventory_data: Dict[str, Any] = None,
        tracking_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive analytics from analysis results.

        Args:
            object_counts: {'workers': int, 'forklifts': int, 'boxes': int, 'pallets': int}
            occupancy_data: Output from OccupancyAnalyzer.
            zone_violations: Output from ZoneDetector.
            inventory_data: Output from InventoryAnalyzer.
            tracking_data: Output from WarehouseTracker (video only).

        Returns:
            Structured analytics dictionary.
        """
        analytics = {
            "object_counts": object_counts,
            "occupancy": occupancy_data.get("occupancy_percent", 0),
            "space_utilization": occupancy_data.get("space_utilization_percent", 0),
            "restricted_zone_violations": zone_violations.get("total_violations", 0),
            "insights": [],
        }

        # Generate insights from actual data
        insights = self._generate_insights(
            object_counts, occupancy_data, zone_violations, inventory_data, tracking_data
        )
        analytics["insights"] = insights

        # Add tracking summary if available
        if tracking_data:
            analytics["tracking"] = {
                "unique_workers": tracking_data.get("unique_workers", 0),
                "unique_forklifts": tracking_data.get("unique_forklifts", 0),
                "unique_boxes": tracking_data.get("unique_boxes", 0),
                "unique_pallets": tracking_data.get("unique_pallets", 0),
                "total_frames": tracking_data.get("total_frames", 0),
                "processed_frames": tracking_data.get("processed_frames", 0),
            }

        # Add inventory data if available
        if inventory_data:
            analytics["inventory"] = inventory_data

        return analytics

    def _generate_insights(
        self,
        counts: Dict[str, int],
        occupancy: Dict[str, Any],
        violations: Dict[str, Any],
        inventory: Optional[Dict[str, Any]],
        tracking: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate meaningful insights based on actual analysis data.
        Only creates insights that are supported by the data.
        """
        insights = []

        total_objects = sum(counts.values())

        if total_objects == 0:
            insights.append("No warehouse objects detected in this analysis.")
            return insights

        # Worker activity
        workers = counts.get("workers", 0)
        if workers > 10:
            insights.append(f"High worker activity detected ({workers} workers visible).")
        elif workers > 0:
            insights.append(f"Worker activity detected ({workers} workers visible).")

        # Forklift activity
        forklifts = counts.get("forklifts", 0)
        if forklifts > 0:
            insights.append(f"Forklift activity detected ({forklifts} forklifts in operation).")

        # Inventory
        boxes = counts.get("boxes", 0)
        pallets = counts.get("pallets", 0)
        if boxes > 0 or pallets > 0:
            parts = []
            if boxes > 0:
                parts.append(f"{boxes} boxes")
            if pallets > 0:
                parts.append(f"{pallets} pallets")
            insights.append(f"Inventory detected: {', '.join(parts)}.")

        # Occupancy
        occ = occupancy.get("occupancy_percent", 0)
        if occ > 70:
            insights.append(f"Warehouse occupancy is high ({occ:.1f}%).")
        elif occ > 40:
            insights.append(f"Warehouse occupancy is moderate ({occ:.1f}%).")
        elif occ > 0:
            insights.append(f"Warehouse occupancy is low ({occ:.1f}%).")

        # Space utilization
        util = occupancy.get("space_utilization_percent", 0)
        if util > 0:
            if util > 50:
                insights.append(f"Space utilization is high ({util:.1f}%).")
            elif util > 20:
                insights.append(f"Space utilization is moderate ({util:.1f}%).")
            else:
                insights.append(f"Space utilization is low ({util:.1f}%).")

        # Zone violations
        viol_count = violations.get("total_violations", 0)
        if viol_count > 0:
            violation_details = violations.get("violations", [])
            classes_in_zone = set(v.get("object_class", "object") for v in violation_details)
            class_str = ", ".join(classes_in_zone)
            insights.append(
                f"⚠️ {viol_count} restricted zone violation(s) detected ({class_str} in restricted area)."
            )

        # Tracking insights
        if tracking:
            unique_w = tracking.get("unique_workers", 0)
            unique_f = tracking.get("unique_forklifts", 0)
            if unique_w > 0:
                insights.append(f"Tracked {unique_w} unique worker(s) across video frames.")
            if unique_f > 0:
                insights.append(f"Tracked {unique_f} unique forklift(s) across video frames.")

        return insights
