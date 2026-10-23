module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
  settings: { react: { version: '18.3' } },
  plugins: ['@typescript-eslint', 'react-hooks', 'jsx-a11y'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:jsx-a11y/recommended',
  ],
  ignorePatterns: ['dist', 'node_modules', 'coverage', '*.cjs'],
  rules: {
    'no-empty': ['error', { allowEmptyCatch: true }],
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',

    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/consistent-type-imports': [
      'error',
      {
        prefer: 'type-imports',
        fixStyle: 'separate-type-imports',
        disallowTypeAnnotations: false,
      },
    ],

    'jsx-a11y/label-has-associated-control': [
      'error',
      { assert: 'either', depth: 3 },
    ],
    'jsx-a11y/no-autofocus': 'error',

    'jsx-a11y/no-redundant-roles': ['error', { ul: ['list'], ol: ['list'] }],

    'jsx-a11y/no-noninteractive-tabindex': [
      'error',
      { tags: [], roles: ['tabpanel', 'region'], allowExpressionValues: true },
    ],

    'no-restricted-syntax': [
      'error',
      {
        selector:
          "CallExpression[callee.property.name='toFixed']",
        message:
          'Use the shared formatter in @/formatting/number (or <NumberValue>) ' +
          'instead of toFixed, so scientific notation and units stay consistent.',
      },
      {
        selector:
          "CallExpression[callee.property.name='toPrecision']",
        message: 'Use formatSignificant from @/formatting/number instead.',
      },
      {
        selector:
          "MemberExpression[object.name='localStorage']",
        message:
          'Use sessionStorage. Nothing in this application should outlive the ' +
          'browser tab, and credentials must never be persisted at all.',
      },
    ],
  },
  overrides: [
    {
      files: ['src/formatting/**/*.ts'],
      rules: { 'no-restricted-syntax': 'off' },
    },
    {
      files: ['tests/**/*.ts', 'tests/**/*.tsx', 'src/**/*.test.ts', 'src/**/*.test.tsx'],
      env: { node: true },
      rules: { 'no-restricted-syntax': 'off' },
    },
  ],
}
