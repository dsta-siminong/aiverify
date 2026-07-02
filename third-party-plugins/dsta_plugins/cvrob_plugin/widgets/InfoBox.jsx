export default function InfoBox({ children }) {
  return (
    <div
      style={{
        width: "100%",
        padding: "16px",
        marginBottom: "24px",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        backgroundColor: "#fafafa",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        breakInside: "avoid",
        pageBreakInside: "avoid",
      }}
    >
      {children}
    </div>
  );
}
