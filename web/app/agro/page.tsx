import { PageHeader } from "@/components/scada/PageHeader";
import { AgroDashboard } from "@/components/agro/AgroDashboard";

export default function AgroPage() {
  return (
    <div>
      <PageHeader tag="AGRO" title="Live Inspection" subtitle="Fruit detection, sizing, and quality grading" />
      <AgroDashboard />
    </div>
  );
}
