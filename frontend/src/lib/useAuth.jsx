import { createContext, useContext } from "react";

export const AuthContext = createContext({ user: null, setUser: () => {} });
export const useAuth = () => useContext(AuthContext);
