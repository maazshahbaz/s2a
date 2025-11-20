import { getServerSession } from "next-auth";

export async function apiPost(path, data) {
  const session = await getServerSession();
  const token = session?.user?.token;

  const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify(data),
    cache: "no-store"
  });

  const json = await res.json();

  if (!res.ok) {
    throw new Error(json.message || "Request failed");
  }

  return json;
}
