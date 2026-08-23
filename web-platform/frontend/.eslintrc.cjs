/**
 * ESLint configuration.
 *
 * The rules that matter most here are the accessibility ones: this is a
 * teaching tool that will be used on school hardware and by people relying on
 * screen readers, and an unlabelled input is a defect, not a style preference.
 *
 * `no-restricted-syntax` enforces the project's own rule that no number reaches
 * the screen without going through the shared formatter - the defect this
 * migration exists to fix.
 */
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
        // The lazy Plotly loader's return type is genuinely `typeof
        // import('plotly.js-dist-min')`; there is no static import to point at,
        // because the whole point is that the module is not statically loaded.
        disallowTypeAnnotations: false,
      },
    ],

    // Accessibility, treated as errors rather than warnings.
    'jsx-a11y/label-has-associated-control': [
      'error',
      { assert: 'either', depth: 3 },
    ],
    'jsx-a11y/no-autofocus': 'error',

    // `role="list"` on a <ul> is redundant in the specification but NOT in
    // practice: Safari with VoiceOver drops list semantics from any list styled
    // `list-style: none`, which every list in this interface is. Removing the
    // role would make those lists unannounced for VoiceOver users, so it is
    // kept deliberately.
    'jsx-a11y/no-redundant-roles': ['error', { ul: ['list'], ol: ['list'] }],

    // A scrollable region must be keyboard-focusable, or a keyboard-only user
    // cannot scroll it (WCAG 2.1.1). Wide result tables are exactly that: a
    // labelled `role="region"` with `overflow-x: auto` and `tabIndex={0}`.
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
      // The formatter is the one place allowed to do primitive number work.
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
