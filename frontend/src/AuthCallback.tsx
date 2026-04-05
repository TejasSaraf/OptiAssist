import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "./supabaseClient";

export default function AuthCallback() {
  const navigate = useNavigate();




useEffect(() => {
  const handleCallback = async () => {
    if (!supabase) {
      navigate("/");
      return;
    }

    // Small delay to let Supabase process the URL hash
    await new Promise((resolve) => setTimeout(resolve, 500));

    const { data, error } = await supabase.auth.getSession();

    if (error || !data.session) {
      console.error("No session found:", error);
      navigate("/");
      return;
    }

    // Save user to DB
    const user = data.session.user;
    const { error: upsertError } = await supabase.from("users").upsert({
      id: user.id,
      email: user.email,
      name: user.user_metadata?.full_name,
      avatar: user.user_metadata?.avatar_url,
      last_login: new Date().toISOString(),
    }, { onConflict: "id" });

    if (upsertError) console.error("DB error:", upsertError);

    navigate("/app");
  };

  handleCallback();
}, [navigate]);

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-8 h-8 border-2 border-green-400 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-gray-500 text-sm">Signing you in...</p>
      </div>
    </div>
  );
}
