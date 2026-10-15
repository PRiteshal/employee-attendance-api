"""
app = FastAPI()
app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan
)
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (real environment variables win over a local .env)


Layout of this file
  1. configuration, database, indexes
  2. time helpers and business rules R1-R5          (pure functions, no I/O)
  3. request validation models
  4. serialisation (BSON document -> API JSON)
  5. query / pipeline builders                      (shared by the endpoints AND /admin/explain)
  6. endpoints
"""
import calendar
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

# override=False (the default): real environment variables take priority over .env
load_dotenv()
log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration, database, indexes
# --------------------------------------------------------------------------- #
# tz_aware=True: PyMongo hands back timezone-aware UTC datetimes instead of naive ones,
# so a stored instant can never be mistaken for server-local time.
client = MongoClient(
    os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
    serverSelectionTimeoutMS=5000,
)
db = client[os.environ.get("MONGO_DB", "attendance_db")]

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]
IDX_LOG_EMP_DATE = "emp_date_unique"
IDX_LOG_DATE_EMP_LATE = "date_emp_late"
IDX_EMP_DEPT_JOINED = "dept_joined"
IDX_EMP_JOINED = "joined_on"

_indexes_ready = False


def ensure_indexes() -> None:
    """Idempotent: create_index is a no-op when the same index already exists."""
    global _indexes_ready
    # unique business key of an employee; also what makes concurrent duplicate creates yield one 201
    db.employees.create_index([("emp_code", 1)], unique=True, name="emp_code_unique")
    # department filters, headcount (department + joined_on <= day), department trend
    db.employees.create_index([("department", 1), ("joined_on", 1)], name=IDX_EMP_DEPT_JOINED)
    # headcount across all departments (joined_on <= end of month)
    db.employees.create_index([("joined_on", 1)], name=IDX_EMP_JOINED)
    # natural key of a record. Unique => race-safe punch-in. Serves: per-employee lists, monthly summary,
    # punch-out lookup, and the $lookup joins from employees.
    db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True, name=IDX_LOG_EMP_DATE)
    # date-range scans: GET /attendance sort (date desc, emp_code asc) and the late leaderboard, which can be
    # answered from the index alone (date range + late_minutes > 0, grouped by emp_code) without touching documents.
    db.attendance_logs.create_index(
        [("date", -1), ("emp_code", 1), ("late_minutes", 1)], name=IDX_LOG_DATE_EMP_LATE
    )
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError as exc:  # keep the process up; /health reports 503 and retries
        log.error("could not create indexes at startup: %s", exc)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}
# --------------------------------------------------------------------------- #
# 2. Time helpers and business rules (R1 - R5)
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def dt_to_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    if value.tzinfo is None:  # defensive: treat naive values as UTC
        value = value.replace(tzinfo=UTC)
    return (value - EPOCH) // timedelta(milliseconds=1)


def now_dt() -> datetime:
    return ms_to_dt(int(time.time() * 1000))


def hm_to_minutes(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return hm_to_minutes(shift_end) <= hm_to_minutes(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: IST calendar date of the punch-in; for overnight shifts a punch-in earlier than shift_end
    belongs to the shift that started the previous day."""
    ist = punch_in.astimezone(IST)
    day = ist.date()
    if is_overnight(shift_start, shift_end):
        seconds_of_day = ist.hour * 3600 + ist.minute * 60 + ist.second
        if seconds_of_day < hm_to_minutes(shift_end) * 60:
            day -= timedelta(days=1)
    return day.isoformat()


def shift_bounds(date_str: str, shift_start: str, shift_end: str) -> Tuple[datetime, datetime]:
    """Shift start/end instants (IST) for the shift that belongs to attendance date `date_str`."""
    base = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=IST)
    start = base + timedelta(minutes=hm_to_minutes(shift_start))
    end = base + timedelta(minutes=hm_to_minutes(shift_end))
    if is_overnight(shift_start, shift_end):
        end += timedelta(days=1)
    return start, end


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then floor(minutes since shift start)."""
    start, _ = shift_bounds(date_str, shift_start, shift_end)
    seconds = int((punch_in - start).total_seconds())
    return seconds // 60 if seconds > 600 else 0


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: floor(minutes) from shift end to punch-out, counted only if >= 30."""
    _, end = shift_bounds(date_str, shift_start, shift_end)
    seconds = int((punch_out - end).total_seconds())
    minutes = seconds // 60 if seconds > 0 else 0
    return minutes if minutes >= 30 else 0


def round_half_up(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, 2 decimals, half-up (Decimal, so no float or banker's rounding surprises)."""
    seconds = int((punch_out - punch_in).total_seconds())
    return float(round_half_up(Decimal(seconds) / Decimal(3600), 2))


def derive(emp: dict, date_str: str, status: str, punch_in: Optional[datetime],
           punch_out: Optional[datetime]) -> dict:
    """All four stored derived fields for a record (see the table in DATA_MODEL.md section 3)."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    ss, se = emp["shift_start"], emp["shift_end"]
    out = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, date_str, ss, se),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        out["work_hours"] = compute_work_hours(punch_in, punch_out)
        out["overtime_minutes"] = compute_overtime(punch_out, date_str, ss, se)
        out["half_day"] = Decimal(str(out["work_hours"])) < Decimal("4.50")  # R5, on the rounded value
    return out


# --------------------------------------------------------------------------- #
# 3. Request models and parameter validation
# --------------------------------------------------------------------------- #
# strict=True rejects floats, strings and booleans; the bounds reject seconds-looking values (422).
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
HM = r"^([01]\d|2[0-3]):[0-5]\d$"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"


def invalid(where: str, field: str, msg: str) -> RequestValidationError:
    """A 422 in FastAPI's own error shape, for checks that cannot be expressed in the schema."""
    return RequestValidationError([{"type": "value_error", "loc": (where, field), "msg": msg, "input": None}])


def parse_date(value: str, field: str, where: str = "query") -> Date:
    if not DATE_RE.match(value):
        raise invalid(where, field, "must be YYYY-MM-DD")
    try:
        return Date.fromisoformat(value)
    except ValueError:
        raise invalid(where, field, "not a valid calendar date")


def month_bounds(month: str, field: str = "month") -> Tuple[str, str]:
    if not re.match(MONTH_PATTERN, month):
        raise invalid("query", field, "must be YYYY-MM")
    year, mon = int(month[:4]), int(month[5:])
    try:
        last = calendar.monthrange(year, mon)[1]
    except ValueError:
        raise invalid("query", field, "invalid month")
    return f"{month}-01", f"{month}-{last:02d}"


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HM)
    shift_end: str = Field(default="18:30", pattern=HM)
    joined_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def _check(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        try:
            Date.fromisoformat(self.joined_on)
        except ValueError:
            raise ValueError("joined_on is not a valid calendar date")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _no_explicit_null(self):
        # "omitted" means unchanged; an explicit null is not allowed by the contract
        for name in ("status", "punch_in", "punch_out"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} must not be null")
        return self


# --------------------------------------------------------------------------- #
# 4. Serialisation (BSON -> API)
# --------------------------------------------------------------------------- #
def _api_value(value: Any) -> Any:
    return dt_to_ms(value) if isinstance(value, datetime) else value


def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc.get("name"),
        "email": doc.get("email"),
        "department": doc.get("department"),
        "shift_start": doc.get("shift_start"),
        "shift_end": doc.get("shift_end"),
        "joined_on": doc.get("joined_on"),
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def serialize_record(doc: dict) -> dict:
    """Never exposes _id. Missing history / half_day / derived values get their documented defaults."""
    history = []
    for entry in doc.get("history") or []:
        history.append({
            "at": dt_to_ms(entry.get("at")),
            "by": entry.get("by"),
            "reason": entry.get("reason"),
            "changes": {
                key: {"from": _api_value(change.get("from")), "to": _api_value(change.get("to"))}
                for key, change in (entry.get("changes") or {}).items()
            },
        })
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# 5. Query / pipeline builders
#    Each plan_* function returns what to run; the endpoint and /admin/explain both use it, so the
#    explained query is by construction the query the endpoint runs.
# --------------------------------------------------------------------------- #
def plan_attendance_list(emp_code, date_from, date_to, status, page, page_size) -> dict:
    flt: Dict[str, Any] = {}
    if emp_code:
        flt["emp_code"] = emp_code
    if date_from is not None or date_to is not None:
        rng: Dict[str, str] = {}
        if date_from is not None:
            rng["$gte"] = parse_date(date_from, "date_from").isoformat()
        if date_to is not None:
            rng["$lte"] = parse_date(date_to, "date_to").isoformat()
        if "$gte" in rng and "$lte" in rng and rng["$gte"] > rng["$lte"]:
            raise invalid("query", "date_from", "date_from must not be after date_to")
        flt["date"] = rng
    if status:
        flt["status"] = status
    return {
        "collection": "attendance_logs",
        "filter": flt,
        "sort": [("date", -1), ("emp_code", 1)],
        "skip": (page - 1) * page_size,
        "limit": page_size,
        "hint": IDX_LOG_EMP_DATE if emp_code else IDX_LOG_DATE_EMP_LATE,
    }


# --- aggregation expression fragments ---------------------------------------
_IS_WEEKDAY = {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}},
                       [2, 3, 4, 5, 6]]}  # $dayOfWeek: Sunday = 1 ... Saturday = 7
_IS_PRESENT = {"$in": ["$status", PRESENCE]}
_HAS_HOURS = {"$in": [{"$type": "$work_hours"}, ["double", "int", "long", "decimal"]]}
_HALF = {"$eq": [{"$ifNull": ["$half_day", False]}, True]}
_LATE_FLAG = {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
# present-days counted in HALVES (a half day = 1, a full day = 2), Mon-Fri only, so every sum stays an exact integer
_PRESENT_HALVES = {"$cond": [{"$and": [_IS_PRESENT, _IS_WEEKDAY]}, {"$cond": [_HALF, 1, 2]}, 0]}


def _rhu(n: Any, d: Any, tag: str) -> dict:
    """Integer expression for round-half-up(n / d) using exact integer arithmetic:
    floor((2n + d) / (2d)). Avoids $round (banker's rounding) and floating-point ties."""
    a, b = f"a{tag}", f"b{tag}"
    return {"$let": {
        "vars": {a: {"$add": [{"$multiply": [2, n]}, d]}, b: {"$multiply": [2, d]}},
        "in": {"$divide": [{"$subtract": [f"$${a}", {"$mod": [f"$${a}", f"$${b}"]}]}, f"$${b}"]},
    }}


def plan_employee_monthly(emp_code: str, month: str) -> dict:
    start, end = month_bounds(month)
    pipeline = [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {"$group": {
            "_id": None,
            "present_halves": {"$sum": _PRESENT_HALVES},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": _LATE_FLAG},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]
    return {"collection": "attendance_logs", "pipeline": pipeline, "hint": IDX_LOG_EMP_DATE,
            "start": start, "end": end}


def plan_department_summary(month: str, department: Optional[str]) -> dict:
    start, end = month_bounds(month)
    match: Dict[str, Any] = {"joined_on": {"$lte": end}}  # R9
    if department:
        match["department"] = department
    per_employee = [  # runs once per employee inside $lookup, using the (emp_code, date) index
        {"$match": {"date": {"$gte": start, "$lte": end}}},
        {"$group": {
            "_id": None,
            "present_halves": {"$sum": _PRESENT_HALVES},
            "late_count": {"$sum": _LATE_FLAG},
            "late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "leave": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "on_duty": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
            # work hours as integer hundredths so the department mean is exact
            "work_cents": {"$sum": {"$cond": [{"$and": [_IS_PRESENT, _HAS_HOURS]},
                                              {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]}},
            "work_n": {"$sum": {"$cond": [{"$and": [_IS_PRESENT, _HAS_HOURS]}, 1, 0]}},
        }},
    ]

    def first(field: str) -> dict:
        return {"$sum": {"$ifNull": [{"$arrayElemAt": [f"$m.{field}", 0]}, 0]}}

    pipeline = [
        {"$match": match},
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": per_employee, "as": "m"}},
        # starting from employees (not from logs) is what keeps people with zero logs in the headcount
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_halves": first("present_halves"),
            "late_count": first("late_count"),
            "total_late_minutes": first("late_minutes"),
            "leave_count": first("leave"),
            "on_duty_count": first("on_duty"),
            "work_cents": first("work_cents"),
            "work_n": first("work_n"),
        }},
        {"$sort": {"_id": 1}},
    ]
    return {"collection": "employees", "pipeline": pipeline,
            "hint": IDX_EMP_DEPT_JOINED if department else IDX_EMP_JOINED}


def plan_late_leaderboard(month: str, limit: int, department: Optional[str]) -> dict:
    start, end = month_bounds(month)
    pipeline: List[dict] = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"},
                    "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs whose employee does not exist are dropped here
    ]
    if department:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank = standard competition ranking: ties share a rank, the next rank is skipped (1, 2, 2, 4)
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit is applied AFTER ranking
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name",
                      "department": "$emp.department", "total_late_minutes": 1, "late_count": 1}},
    ]
    return {"collection": "attendance_logs", "pipeline": pipeline, "hint": IDX_LOG_DATE_EMP_LATE}


def plan_department_trend(department: str, date_from: str, date_to: str) -> dict:
    d_from, d_to = parse_date(date_from, "from"), parse_date(date_to, "to")
    if d_to < d_from:
        raise invalid("query", "to", "to must not be before from")
    if (d_to - d_from).days + 1 > 92:
        raise invalid("query", "to", "range must not exceed 92 days")
    lo = datetime(d_from.year, d_from.month, d_from.day, tzinfo=UTC)
    hi = datetime(d_to.year, d_to.month, d_to.day, tzinfo=UTC) + timedelta(days=1)  # upper bound is exclusive
    window = {"documents": [-6, 0]}  # current row + up to 6 preceding rows, inside the requested range only
    pipeline = [
        {"$match": {"department": department}},
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": [{"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
                                  {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1,
                                                "late_minutes": 1}}],
                     "as": "logs"}},
        # keep employees without logs: guarantees the stream is never empty for $densify
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": "$logs.date",
            "halves": {"$sum": {"$cond": [{"$in": ["$logs.status", PRESENCE]},
                                          {"$cond": [{"$eq": [{"$ifNull": ["$logs.half_day", False]}, True]}, 1, 2],
                                          }, 0]}},
            "late": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]}, 1, 0]}},
        }},
        {"$addFields": {"day": {"$cond": [{"$eq": [{"$type": "$_id"}, "string"]},
                                          {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}},
                                          lo]}}},  # the "no logs" bucket becomes a harmless zero row on day one
        {"$sort": {"day": 1}},
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [lo, hi]}}},
        {"$group": {"_id": "$day", "halves": {"$sum": "$halves"}, "late": {"$sum": "$late"}}},
        {"$sort": {"_id": 1}},
        {"$addFields": {"date": {"$dateToString": {"date": "$_id", "format": "%Y-%m-%d"}},
                        "dow": {"$dayOfWeek": "$_id"}}},
        # R9 headcount per day: department employees with joined_on <= that day
        {"$lookup": {"from": "employees", "let": {"d": "$date"},
                     "pipeline": [{"$match": {"department": department, "$expr": {"$lte": ["$joined_on", "$$d"]}}},
                                  {"$count": "n"}],
                     "as": "hc"}},
        {"$addFields": {"headcount": {"$ifNull": [{"$arrayElemAt": ["$hc.n", 0]}, 0]},
                        "is_working_day": {"$in": ["$dow", [2, 3, 4, 5, 6]]}}},
        # rate scaled by 10000 and rounded half-up with integer maths: halves / 2 / headcount * 10000
        {"$addFields": {"rate_i": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            _rhu({"$multiply": ["$halves", 10000]}, {"$multiply": [2, "$headcount"]}, "r"),
            None]}}},
        {"$setWindowFields": {"sortBy": {"_id": 1}, "output": {
            "wsum": {"$sum": "$rate_i", "window": window},  # $sum ignores null rates
            "wcnt": {"$sum": {"$cond": [{"$eq": [{"$type": "$rate_i"}, "null"]}, 0, 1]}, "window": window},
        }}},
        {"$addFields": {"mavg_i": {"$cond": [{"$gt": ["$wcnt", 0]}, _rhu("$wsum", "$wcnt", "m"), None]}}},
        {"$project": {"_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "halves": 1, "late": 1,
                      "rate_i": 1, "mavg_i": 1}},
        {"$sort": {"date": 1}},
    ]
    return {"collection": "employees", "pipeline": pipeline, "hint": IDX_EMP_DEPT_JOINED}


def _count_working_days(start: Date, end: Date) -> int:
    if end < start:
        return 0
    n = (end - start).days + 1
    full, rest = divmod(n, 7)
    return full * 5 + sum(1 for i in range(rest) if (start + timedelta(days=i)).weekday() < 5)


# --------------------------------------------------------------------------- #
# 6. Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        if not _indexes_ready:
            ensure_indexes()
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


# ---- employees --------------------------------------------------------------
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    created = now_dt()
    doc["created_at"] = created
    try:
        # no check-then-insert: the unique emp_code index decides, so parallel creates give one 201 and the rest 409
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    flt = {"department": department} if department else {}
    total = db.employees.count_documents(flt)
    cursor = db.employees.find(flt).sort("emp_code", 1).skip((page - 1) * page_size).limit(page_size)
    return {"items": [serialize_employee(d) for d in cursor], "total": total, "page": page,
            "page_size": page_size}


# ---- attendance -------------------------------------------------------------
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    date_str = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": date_str,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, date_str, emp["shift_start"], emp["shift_end"]),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        # the unique (emp_code, date) index arbitrates concurrent punch-ins: exactly one insert wins
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return serialize_record(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    # most recent record whose punch_in <= punched_at. Looking up by punch_in (not by the date of the punch-out)
    # is what makes this work across midnight. ABSENT/LEAVE have no punch_in and never match.
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("date", -1)]
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    punch_in_dt = rec["punch_in"]
    if ts <= punch_in_dt:
        raise invalid("body", "punched_at", "punched_at must be after punch_in")
    if ts - punch_in_dt > timedelta(hours=24):
        raise invalid("body", "punched_at", "punch-out is more than 24 hours after punch-in")
    derived = derive(emp, rec["date"], rec["status"], punch_in_dt, ts)
    # compare-and-set: only closes the record if it is still open and unchanged, so of two simultaneous
    # punch-outs exactly one matches (the loser gets None and therefore 409)
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None, "punch_in": punch_in_dt},
        {"$set": {"punch_out": ts, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return serialize_record(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    plan = plan_attendance_list(emp_code, date_from, date_to, status, page, page_size)
    col = db[plan["collection"]]
    total = col.count_documents(plan["filter"], hint=plan["hint"])
    cursor = (col.find(plan["filter"], hint=plan["hint"]).sort(plan["sort"])
              .skip(plan["skip"]).limit(plan["limit"]))
    return {"items": [serialize_record(d) for d in cursor], "total": total, "page": page,
            "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, body: RegularizeIn, date: str = Path(...)):
    parse_date(date, "date", "path")
    emp = get_employee_or_404(emp_code)
    for _attempt in range(5):  # optimistic concurrency: re-read and retry if someone else wrote in between
        rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
        if rec is None:
            raise HTTPException(404, "no record for that date")
        old_hist = rec.get("history") or []
        old = {
            "status": rec["status"],
            "punch_in": rec.get("punch_in"),
            "punch_out": rec.get("punch_out"),
            "work_hours": rec.get("work_hours"),
            "late_minutes": rec.get("late_minutes") or 0,
            "overtime_minutes": rec.get("overtime_minutes") or 0,
            "half_day": bool(rec.get("half_day", False)),
        }
        sent = body.model_fields_set
        new_status = body.status or old["status"]

        if new_status in ("ABSENT", "LEAVE"):
            if "punch_in" in sent or "punch_out" in sent:
                raise invalid("body", "status", "ABSENT/LEAVE clear the punch times; do not send punch_in/punch_out")
            new_in = new_out = None
        else:
            new_in = ms_to_dt(body.punch_in) if "punch_in" in sent else old["punch_in"]
            new_out = ms_to_dt(body.punch_out) if "punch_out" in sent else old["punch_out"]
            if new_in is None:
                raise invalid("body", "punch_in", "a presence status requires a punch_in")
            if attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:
                raise invalid("body", "punch_in", "punch_in must stay on the record's attendance date")
            if new_out is not None and not (new_in < new_out <= new_in + timedelta(hours=24)):
                raise invalid("body", "punch_out", "punch_out must be after punch_in and within 24 hours")

        new = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
               **derive(emp, date, new_status, new_in, new_out)}
        if all(new[k] == old[k] for k in ("status", "punch_in", "punch_out")):
            raise invalid("body", "status", "the request changes nothing")

        entry = {
            "at": now_dt(),
            "by": body.regularized_by,
            "reason": body.reason,
            "changes": {k: {"from": old[k], "to": new[k]} for k in old if old[k] != new[k]},
        }
        # the filter pins the exact state we computed from (including the history length): if another writer
        # got in first nothing matches, we re-read and recompute instead of overwriting or losing its entry
        updated = db.attendance_logs.find_one_and_update(
            {"emp_code": emp_code, "date": date, "status": old["status"],
             "punch_in": old["punch_in"], "punch_out": old["punch_out"],
             "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(old_hist)]}},
            {"$set": new, "$push": {"history": entry}},
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return serialize_record(updated)
    raise HTTPException(409, "record was modified concurrently, please retry")


# ---- analytics --------------------------------------------------------------
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=MONTH_PATTERN)):
    emp = get_employee_or_404(emp_code)
    plan = plan_employee_monthly(emp_code, month)
    row = next(iter(db[plan["collection"]].aggregate(plan["pipeline"], hint=plan["hint"])), {})
    start, end = Date.fromisoformat(plan["start"]), Date.fromisoformat(plan["end"])
    joined = Date.fromisoformat(emp["joined_on"])
    working_days = _count_working_days(max(start, joined), end)  # R7: counted from joined_on
    halves = int(row.get("present_halves", 0))
    present_days = Decimal(halves) / 2
    pct = None
    if working_days > 0:
        pct = float(round_half_up(present_days / working_days * 100, 2))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": float(present_days),
        "leave_days": int(row.get("leave_days", 0)),
        "late_count": int(row.get("late_count", 0)),
        "total_late_minutes": int(row.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(row.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(..., pattern=MONTH_PATTERN), department: Optional[str] = None):
    plan = plan_department_summary(month, department)
    items = []
    for row in db[plan["collection"]].aggregate(plan["pipeline"], hint=plan["hint"]):
        n = int(row["work_n"])
        avg = None
        if n > 0:  # mean over records, not a mean of per-employee means
            avg = float(round_half_up(Decimal(int(round(row["work_cents"]))) / Decimal(n) / 100, 2))
        items.append({
            "department": row["_id"],
            "headcount": int(row["headcount"]),
            "present_days": float(Decimal(int(row["present_halves"])) / 2),
            "avg_work_hours": avg,
            "late_count": int(row["late_count"]),
            "total_late_minutes": int(row["total_late_minutes"]),
            "leave_count": int(row["leave_count"]),
            "on_duty_count": int(row["on_duty_count"]),
        })
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str = Query(..., pattern=MONTH_PATTERN), limit: int = Query(10, ge=1, le=50),
                     department: Optional[str] = None):
    plan = plan_late_leaderboard(month, limit, department)
    items = [
        {"rank": int(r["rank"]), "emp_code": r["emp_code"], "name": r["name"], "department": r["department"],
         "total_late_minutes": int(r["total_late_minutes"]), "late_count": int(r["late_count"])}
        for r in db[plan["collection"]].aggregate(plan["pipeline"], hint=plan["hint"])
    ]
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    plan = plan_department_trend(department, from_, to)  # validates the range first (422)
    if db.employees.count_documents({"department": department}, limit=1) == 0:
        raise HTTPException(404, "unknown department")
    items = []
    for r in db[plan["collection"]].aggregate(plan["pipeline"], hint=plan["hint"]):
        rate = None if r["rate_i"] is None else float(Decimal(int(r["rate_i"])).scaleb(-4))
        mavg = None if r["mavg_i"] is None else float(Decimal(int(r["mavg_i"])).scaleb(-4))
        items.append({
            "date": r["date"],
            "is_working_day": bool(r["is_working_day"]),
            "headcount": int(r["headcount"]),
            "present_count": float(Decimal(int(r["halves"])) / 2),
            "late_count": int(r["late"]),
            "attendance_rate": rate,
            "moving_avg_7d": mavg,
        })
    return {"department": department, "items": items}


# ---- admin: explain ---------------------------------------------------------
EXPLAIN_ENDPOINTS = ("attendance_list", "employee_monthly", "department_summary", "late_leaderboard",
                     "department_trend")


def _require(**params):
    for name, value in params.items():
        if value is None:
            raise invalid("query", name.rstrip("_"), "field required")


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_PATTERN),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint not in EXPLAIN_ENDPOINTS:
        raise invalid("path", "endpoint", f"must be one of {', '.join(EXPLAIN_ENDPOINTS)}")

    if endpoint == "attendance_list":
        plan = plan_attendance_list(emp_code, date_from, date_to, status, page, page_size)
        command: Dict[str, Any] = {"find": plan["collection"], "filter": plan["filter"],
                                   "sort": dict(plan["sort"]), "limit": plan["limit"], "hint": plan["hint"]}
        if plan["skip"]:
            command["skip"] = plan["skip"]
    else:
        if endpoint == "employee_monthly":
            _require(emp_code=emp_code, month=month)
            plan = plan_employee_monthly(emp_code, month)
        elif endpoint == "department_summary":
            _require(month=month)
            plan = plan_department_summary(month, department)
        elif endpoint == "late_leaderboard":
            _require(month=month)
            plan = plan_late_leaderboard(month, limit, department)
        else:
            _require(department=department, from_=from_, to=to)
            plan = plan_department_trend(department, from_, to)
        command = {"aggregate": plan["collection"], "pipeline": plan["pipeline"], "cursor": {},
                   "hint": plan["hint"]}

    raw = db.command("explain", command, verbosity="executionStats")
    return {
        "endpoint": endpoint,
        "collection": plan["collection"],
        # BSON -> plain JSON (dates, Int64, Decimal128 ...)
        "explain": json.loads(json_util.dumps(raw, json_options=json_util.RELAXED_JSON_OPTIONS)),
    }
