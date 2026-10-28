import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Face ID Biometric Security System | XGBoost & Random Forest",
  description: "Enterprise Biometric Access Control Dashboard powered by XGBoost, Random Forest, and OpenCV Liveness Detection",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-cyber-dark text-slate-100 antialiased selection:bg-cyan-500 selection:text-black">
        {children}
      </body>
    </html>
  );
}
