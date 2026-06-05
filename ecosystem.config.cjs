module.exports = {
  apps: [{
    name: 'protacpred',
    interpreter: 'python3',
    script: 'app.py',
    cwd: '/home/user/SE3AF_work/SE3AF_v381_FIXED',
    env: {
      FLASK_APP: 'app.py',
      PORT: 5000
    },
    watch: false,
    instances: 1,
    exec_mode: 'fork'
  }]
}
