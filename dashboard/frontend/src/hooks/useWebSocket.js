import { useState, useRef, useCallback } from 'react';

const useWebSocket = () => {
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const listenersRef = useRef({});

  const connect = useCallback(() => {
    try {
      // Try to connect to WebSocket if available
      // const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      
      // For now, simulate connection for demo purposes
      setConnected(true);
      console.log('WebSocket connection established (simulated)');
    } catch (error) {
      console.log('WebSocket connection disabled to prevent errors');
      setConnected(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      setConnected(false);
    }
  }, []);

  const on = useCallback((event, callback) => {
    if (socketRef.current) {
      socketRef.current.on(event, callback);
      if (!listenersRef.current[event]) {
        listenersRef.current[event] = [];
      }
      listenersRef.current[event].push(callback);
    }
  }, []);

  const off = useCallback((event, callback) => {
    if (socketRef.current) {
      socketRef.current.off(event, callback);
      if (listenersRef.current[event]) {
        listenersRef.current[event] = listenersRef.current[event].filter(
          (cb) => cb !== callback
        );
      }
    }
  }, []);

  const emit = useCallback((event, data) => {
    if (socketRef.current && connected) {
      socketRef.current.emit(event, data);
    }
  }, [connected]);

  return {
    connected,
    connect,
    disconnect,
    on,
    off,
    emit,
  };
};

export { useWebSocket };
export default useWebSocket;
