import { useNavigate } from "react-router-dom";
import Dashboard from "./components/dashboard";

export default function AppPage() {
  const navigate = useNavigate();

  const handleReturnHome = () => {
    navigate("/");
  };

  return <Dashboard onBackHome={handleReturnHome} />;
}
