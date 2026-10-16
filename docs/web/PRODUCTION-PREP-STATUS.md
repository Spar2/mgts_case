# Public demo and team production preparation

Локальная приёмка завершена 11 октября 2026 года. Результаты и границы проверки:
[PRODUCTION-ACCEPTANCE.md](PRODUCTION-ACCEPTANCE.md).

- [x] Additive database migrations; invitations and password recovery.
- [x] Shared Redis limits; password and session management; idle expiry.
- [x] Immutable bounded real-data public package and isolated public APIs.
- [x] Public landing, try, access request, activation, recovery and account UI.
- [x] Project job capacity, production HTTPS/Host/Origin/proxy configuration.
- [x] Route-specific Nginx body limits verified through the proxy.
- [x] Encrypted backups, seven-day retention and empty-database restore.
- [x] Operational metrics and safe logs.
- [x] Security, concurrency, outage and scientific-boundary tests: 44 passed.
- [x] Chromium, Firefox, WebKit and 320–430 px user journeys.
- [x] 30-minute mixed load with five public and five authenticated users.
- [x] Participant/operator instructions, permission matrix and measured report.
- [x] Local stand updated, checks passing and changes prepared on `web`.

No external deployment, OAuth or arbitrary CSV uploads are part of this stage.
Physical phones, external networking and an independent security audit remain
outside the measured evidence. External publication requires real certificates,
secrets and off-server backups.
