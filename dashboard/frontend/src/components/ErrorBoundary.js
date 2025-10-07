import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Check if it's a MetaMask or browser extension error
    if (error && error.message && (
      error.message.includes('MetaMask') ||
      error.message.includes('chrome-extension') ||
      error.message.includes('Failed to connect to MetaMask') ||
      error.message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
      error.message.includes('inpage.js') ||
      error.message.includes('ethereum') ||
      error.message.includes('web3')
    )) {
      // Don't show error for browser extension issues
      console.warn('Browser extension error ignored:', error.message);
      return { hasError: false, error: null };
    }
    
    // Also check error stack trace
    if (error && error.stack && (
      error.stack.includes('chrome-extension') ||
      error.stack.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
      error.stack.includes('inpage.js')
    )) {
      console.warn('Browser extension error ignored (from stack):', error.message);
      return { hasError: false, error: null };
    }
    
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error but don't show it if it's a browser extension issue
    if (error && error.message && (
      error.message.includes('MetaMask') ||
      error.message.includes('chrome-extension') ||
      error.message.includes('Failed to connect to MetaMask') ||
      error.message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
      error.message.includes('inpage.js') ||
      error.message.includes('ethereum') ||
      error.message.includes('web3')
    )) {
      console.warn('Browser extension error caught and ignored:', error.message);
      return;
    }
    
    // Also check error stack trace
    if (error && error.stack && (
      error.stack.includes('chrome-extension') ||
      error.stack.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
      error.stack.includes('inpage.js')
    )) {
      console.warn('Browser extension error caught and ignored (from stack):', error.message);
      return;
    }
    
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
          <div className="max-w-md w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0">
                <svg className="h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Application Error
                </h3>
              </div>
            </div>
            <div className="mt-2">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Something went wrong. Please refresh the page to continue.
              </p>
            </div>
            <div className="mt-4">
              <button
                onClick={() => window.location.reload()}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
              >
                Refresh Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
