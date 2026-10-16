# LAN-стенд: запуск на дев-ВМ (192.168.1.20)

Всё ниже выполняется на дев-ВМ. С этой машины — только `scp`/`ssh`.

## 1. Доставка кода и данных (с рабочей машины) — legacy дев-ВМ

Канон — git+CI (см. `DATA_GUIDE.md` §1); ниже — legacy scp-доставка только для дев-ВМ.

```powershell
scp stack/compose.yaml stack/CONTRACTS.md gritsko@192.168.1.20:~/haklolarp-stack/
scp stack/clickhouse/init.sql gritsko@192.168.1.20:~/haklolarp-stack/clickhouse/
scp stack/backend/app.py gritsko@192.168.1.20:~/haklolarp-stack/backend/
scp stack/streamer/stream.py gritsko@192.168.1.20:~/haklolarp-stack/streamer/
scp stack/registry/manifest.json gritsko@192.168.1.20:~/haklolarp-stack/registry/
scp ../1.csv ../2.csv gritsko@192.168.1.20:~/haklolarp-stack/data/
```

Каталоги создать заранее:
`mkdir -p ~/haklolarp-stack/{clickhouse,backend,streamer,registry,data}`.

Данные — только `1.csv` (горячий) + `2.csv` (контроль) для старта,
остальные по мере места на диске.

## 2. Старт (на дев-ВМ)

```bash
cd ~/haklolarp-stack  # на Xeon — /opt/haklolarp-stack
docker compose -f stack/compose.yaml up -d --build
docker compose -f stack/compose.yaml ps
curl 127.0.0.1:8000/api/manifest
```

## 3. Прогон стримера (первые 50 профилей 2.csv)

```bash
docker compose -f stack/compose.yaml run --rm streamer \
  --backend http://backend:8000 --csv /data/2.csv --limit 50 --delay 0.1
```
(На дев-ВМ CSV лежат в `~/haklolarp-stack/data`. На Xeon каталог —
`/opt/haklolarp-stack/data`, это `DATA_DIR` из `stack/.env`, не `./data`
рядом с `compose.yaml`.)

Автомат шагов §2–§3: `bash stack/demo.sh` из корня клона (полный прогон — `FULL=1`).

## 4. Проверка ClickHouse и стримов

```bash
curl -s '127.0.0.1:8123/?query=SELECT%20count()%20FROM%20haklolarp.profiles_raw'
python3 verify.py streams
```

Ожидание: `50` (smoke) и `pending=0 lag=0`. Ошибки валидации — в выводе
стримера с номером строки. `queued` в ответе `/api/infer` = приём в стрим;
в CH пишет ingester (с отставанием в секунды).

Сводка по записям/сегментам: `python3 verify.py report`
(файл `stack/verify.py`, гонять из корня стека на том хосте, где стенд).

## 5. Бэкап (на хосте стенда; STACK — корень стека:
## дев-ВМ `~/haklolarp-stack`, Xeon `/opt/haklolarp-stack`)

```bash
docker run --rm -v haklolarp-stack_ch-data:/vol -v ~/haklo-backup:/b \
  ubuntu tar czf /b/ch-data-$(date +%F).tgz -C /vol .
cp -r $STACK/registry ~/haklo-backup/registry-$(date +%F)
```

## 6. LAN git-remote (обмен кодом без GitHub)

На дев-ВМ один раз:
`git init --bare ~/git/haklolarp.git`

У каждого в клоне:
`git remote add lan gritsko@192.168.1.20:git/haklolarp.git`
`git push lan devops` / `git fetch lan`.
