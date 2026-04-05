import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {supabase} from "./supabaseClient";

interface User {
  name: string;
  email: string;
  avatar: string;
}

export default function AppPage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const getUser = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session) {
        navigate("/");
        return;
      }

      const u = session.user;
      setUser({
        name: u.user_metadata?.full_name || "User",
        email: u.email || "",
        avatar: u.user_metadata?.avatar_url || "",
      });
      setLoading(false);
    };

    getUser();
  }, [navigate]);

  const handleLogout = async () => {
    try {
      // 1. Sign out from Supabase (invalidates server session)
      await supabase.auth.signOut({ scope: "global" });

      // 2. Clear ALL supabase keys from localStorage manually
      Object.keys(localStorage).forEach((key) => {
        if (key.startsWith("sb-")) {
          localStorage.removeItem(key);
        }
      });

      // 3. Navigate to landing page
      navigate("/");
    } catch (error) {
      console.error("Logout error:", error);
      // Even if signOut fails, clear storage and redirect
      Object.keys(localStorage).forEach((key) => {
        if (key.startsWith("sb-")) {
          localStorage.removeItem(key);
        }
      });
      navigate("/");
    }
  };

  if (loading)
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
      </div>
    );

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      <div className="text-center space-y-6">
        {user?.avatar && (
          <img
            src={user.avatar}
            className="w-20 h-20 rounded-full mx-auto border-2 border-green-500"
          />
        )}
        <h1 className="text-4xl font-black">
          Hello, <span className="text-green-400">{user?.name}</span> 👋
        </h1>
        <p className="text-gray-400">{user?.email}</p>
        <p className="text-2xl font-bold text-green-400">
          This is the actual app!
        </p>
        <button
          onClick={handleLogout}
          className="px-6 py-3 rounded-xl border border-red-500/50 text-red-400 hover:bg-red-500/10 transition-all"
        >
          Logout
        </button>
      </div>
    </div>
  );
}