import { PageHeader } from "@/components/scada/PageHeader";
import { IndustDashboard } from "@/components/indust/IndustDashboard";

export default function IndustPage() {
  return (
    <div>
      <PageHeader tag="INDUST" title="Live Inspection" subtitle="PaDiM anomaly detection on industrial parts" />
      <IndustDashboard />
    </div>
  );
}
