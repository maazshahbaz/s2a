"use client";

import { SessionProvider } from "next-auth/react";

export function Providers({ children, serverSession }) {
  return <SessionProvider session={serverSession}>{children}</SessionProvider>;
}
