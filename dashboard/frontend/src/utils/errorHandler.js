/**
 * Global Error Handler for Telecom AI 4.0
 * Handles browser extension errors and other runtime errors
 */

// Store original error handlers
const originalConsoleError = console.error;
const originalWindowError = window.onerror;

// Filter out browser extension errors
const isBrowserExtensionError = (error) => {
  if (typeof error === 'string') {
    return error.includes('MetaMask') || 
           error.includes('chrome-extension') || 
           error.includes('moz-extension') ||
           error.includes('Failed to connect to MetaMask') ||
           error.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
           error.includes('inpage.js');
  }
  
  if (error && error.message) {
    return error.message.includes('MetaMask') || 
           error.message.includes('chrome-extension') || 
           error.message.includes('moz-extension') ||
           error.message.includes('Failed to connect to MetaMask') ||
           error.message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
           error.message.includes('inpage.js');
  }
  
  return false;
};

// Override console.error to filter extension errors
console.error = (...args) => {
  const errorMessage = args.join(' ');
  if (!isBrowserExtensionError(errorMessage)) {
    originalConsoleError.apply(console, args);
  } else {
    // Log as warning instead of error for extension issues
    console.warn('Browser extension error (ignored):', ...args);
  }
};

// Override window.onerror to filter extension errors
window.onerror = (message, source, lineno, colno, error) => {
  if (!isBrowserExtensionError(message) && !isBrowserExtensionError(error)) {
    if (originalWindowError) {
      return originalWindowError(message, source, lineno, colno, error);
    }
    return false;
  } else {
    // Log as warning for extension issues
    console.warn('Browser extension error (ignored):', message, source, lineno, colno, error);
    return true; // Prevent default error handling
  }
};

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  const error = event.reason;
  if (isBrowserExtensionError(error)) {
    console.warn('Browser extension promise rejection (ignored):', error);
    event.preventDefault(); // Prevent default error handling
  }
});

// Initialize error handling
export const initializeErrorHandling = () => {
  console.log('🛡️ Error handling initialized - browser extension errors will be filtered');
  
  // Override window.addEventListener to catch MetaMask errors early
  const originalAddEventListener = window.addEventListener;
  window.addEventListener = function(type, listener, options) {
    if (type === 'error' || type === 'unhandledrejection') {
      const wrappedListener = function(event) {
        if (isBrowserExtensionError(event.error || event.reason || event.message)) {
          console.warn('Browser extension error suppressed:', event.error || event.reason || event.message);
          event.preventDefault();
          event.stopPropagation();
          return false;
        }
        return listener.call(this, event);
      };
      return originalAddEventListener.call(this, type, wrappedListener, options);
    }
    return originalAddEventListener.call(this, type, listener, options);
  };
  
  // Also override the global error handler more aggressively
  window.addEventListener('error', function(event) {
    if (isBrowserExtensionError(event.error || event.message)) {
      console.warn('Global error handler: Browser extension error suppressed');
      event.preventDefault();
      event.stopPropagation();
      return false;
    }
  }, true);
  
  window.addEventListener('unhandledrejection', function(event) {
    if (isBrowserExtensionError(event.reason)) {
      console.warn('Global rejection handler: Browser extension error suppressed');
      event.preventDefault();
      return false;
    }
  }, true);
};

const errorHandler = {
  initializeErrorHandling,
  isBrowserExtensionError
};

export default errorHandler;
