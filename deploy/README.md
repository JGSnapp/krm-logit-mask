# GPU в immers.cloud: свой образ и запуск экспериментов

Один раз подготовить VM (проект, окружение, модели в кэше), снять снимок и
дальше арендовать VM из него: старт без установки и скачивания. Аренду ведёт
live-копия `../../gpu-harbor-live` (demo-копия `../../gpu-harbor` не трогается).

## Что уже настроено в gpu-harbor-live

| | |
|---|---|
| Тариф | `teslat4-1.4.16.120`: Tesla T4 16 ГБ, 4 vCPU, 16 ГБ ОЗУ, диск 120 ГБ — 22.34 ₽/ч (цена с сайта 19.11.2026, подтверждена до 22.11.2026) |
| Образ | Ubuntu 24.04 CUDA (Sep 2026) [BIOS] `08577144-…`; если не загрузится — [UEFI] `9c8faacb-…`. Образ весит 43 ГБ, поэтому диск 120, а не 60 |
| Сеть | `immers` `dd7b6688-…` (публичные 195.209.x) |
| Лимиты | 25 ₽/ч, 300 ₽/сутки, 300 ₽ всего, 150 мин на аренду |

`tools/cloudop.py` (оператор): `show` — статус и IP, `hostkey` — ключи хоста
из консоли VM в `certs/known_hosts`, `console`, `snapshot` — свой образ,
`image`. Удаляет VM только gpu-harbor.

## 1. Первая аренда: подготовка, проверка, снимок

```powershell
cd ..\..\gpu-harbor-live; .\run.ps1          # панель http://127.0.0.1:8787
# запрос t4-16gb на 150 мин -> одобрить -> server_id в статусе
python tools\cloudop.py show <server_id>     # IP
python tools\cloudop.py hostkey <server_id>  # после загрузки
```

Пользователя `trainer` в публичном образе нет, поэтому подготовка идёт
напрямую по SSH под пользователем образа (`ubuntu`, видно в `console`):

```powershell
cd ..\krm-logit-mask; .\deploy\pack.ps1
$key = "..\gpu-harbor-live\certs\aitry-JgSnapp.pem"; $kh = "..\gpu-harbor-live\certs\known_hosts"
scp -i $key -o UserKnownHostsFile=$kh "$env:TEMP\krm-logit-mask.tar.gz" deploy\bootstrap.sh deploy\prepare_snapshot.sh ubuntu@<IP>:/tmp/
ssh -i $key -o UserKnownHostsFile=$kh ubuntu@<IP> "sudo bash /tmp/bootstrap.sh /tmp/krm-logit-mask.tar.gz"
ssh -i $key -o UserKnownHostsFile=$kh trainer@<IP> "cd krm-logit-mask && nohup bash deploy/run_exp29.sh > exp29.log 2>&1 &"
```

`bootstrap.sh`: пользователь `trainer` (без sudo, тот же ключ), uv, проект с
`--extra gpu web lemma dev`, Qwen 2.5 1.5B и 3B в `~/.cache/huggingface`,
проверка CUDA и тесты. Медленно качается с HF — `HF_ENDPOINT=<зеркало>`.

Результаты забрать до конца аренды (VM удаляется вместе с диском):

```powershell
scp -i $key -o UserKnownHostsFile=$kh "trainer@<IP>:krm-logit-mask/experiments/results/exp29_*.json" experiments\results\
```

Снимок — после проверки, что всё работает:

```powershell
ssh ... ubuntu@<IP> "sudo bash /tmp/prepare_snapshot.sh"           # чистка + poweroff
python tools\cloudop.py snapshot <server_id> krm-t4-qwen-2026-11     # ждёт ACTIVE, печатает image_id
```

Потом завершить аренду в панели и записать `image_id` в
`certs/config.json` → `openstack.image_id`. Хранение снимка платное.

## 2. Следующие аренды из своего образа

Аренда → `show` → `hostkey` (ключ хоста у каждой VM новый) → задание через
gpu-harbor или `ssh trainer@<IP>`:

```
cd ~/krm-logit-mask && nohup bash deploy/run_exp29.sh > exp29.log 2>&1 &   # FULL=1 — полная сетка
```

Новый код без пересборки образа: `pack.ps1` → `scp` архива →
`tar -xzf … -C ~/krm-logit-mask` (окружение и модели остаются).
