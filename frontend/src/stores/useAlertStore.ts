import { create } from 'zustand';
export const useAlertStore = create<any>(() => ({ alerts: [] }));
