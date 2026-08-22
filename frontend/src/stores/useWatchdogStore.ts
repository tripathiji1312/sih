import { create } from 'zustand';
export const useWatchdogStore = create<any>(() => ({ status: 'HEALTHY' }));
