# Future vision: seamless, transfer-at-speed transport

MPEE is built to solve today's fleet problems — delivery routing, field
service, waste collection — on a laptop, offline, fast. But the engine is
deliberately rigged for something larger: a future where autonomous cars,
buses, trains and boats hand passengers and goods between each other
**without anyone stopping**.

## The core idea: car convenience at bus economics

A private car gives you A-to-anywhere, door to door, no waiting. A bus or a
train gives you cost-efficiency a car can never touch: one driver (or none),
one drivetrain, hundreds of passengers, near-continuous motion. Today you
must pick one.

The unlock is **transfer at speed**. If a small autonomous pod can dock with
a bus-class vehicle while both are moving — match velocity, latch, exchange
passengers or cargo, detach — then a journey can be:

1. A pod picks you up at your door (last mile, the only individual leg).
2. It merges with a trunk vehicle running 95 % of the distance non-stop.
3. Near your destination, a pod detaches and takes you the final kilometre.

You experience a single uninterrupted ride from A to Å with no queue, no
parking, no timetable. The system experiences bus/train economics: 100
individual Teslas can never economically compete with one trunk vehicle
covering 95 % of the distance without a single stop — the pods only exist at
the edges, where individuality has value.

## On the water: hydrofoils that never come down

The same principle is even more dramatic at sea, because for a hydrofoil the
expensive thing is not moving — it is *stopping*. Foiling drag is a fraction
of hull drag, but every docking forces the vessel down off its foils, into
the high-drag regime, through a harbour manoeuvre, and back up again. The
energy and time cost of each stop dominates the route.

So the foil should **never come down**:

- A large hydrofoil runs continuously along a trunk corridor — up and down
  the Oslofjord, along the European coast, the US seaboard — never docking.
- Smaller feeder boats run out from shore, match speed, dock *while both are
  foiling*, exchange passengers/cargo, and peel off toward land.
- At dedicated quays, the feeder boats interface with the land system: pods,
  buses, trains — the same transfer choreography, one medium over.

A coastline becomes a high-frequency transit line with no stations on the
trunk — every "station" is a moving rendezvous.

## Why this is a routing problem

Transfer-at-speed turns transport into a continuous, rolling optimization
problem that is brutally harder than today's static VRP:

- **Synchronised rendezvous**: a pod and a trunk vehicle must arrive at the
  same point in space *and* time, within seconds — every transfer is a
  time-window constraint two orders of magnitude tighter than a delivery
  slot, and every replan moves thousands of them at once.
- **Fleet-scale matrices, continuously refreshed**: matching thousands of
  pods to trunk vehicles needs the full travel-time matrix of a metro area,
  recomputed as conditions change — exactly the 50k×50k-in-30-seconds,
  500 MB-budget regime MPEE is built for.
- **Re-optimization in seconds, not minutes**: a missed rendezvous cascades.
  The solver that fixes it must produce a near-optimal global replan inside
  the time it takes a pod to reach the next merge point. MPEE's
  solve-quality-per-second — SOTA-class answers in seconds on a laptop —
  is the property that matters here, more than peak quality at infinite
  budget.
- **Multi-leg journeys as first-class constraints**: precedence, k-of-N
  alternatives, transfer chains — the constraint surface MPEE already
  exposes (precedence, disjunctions, custom dimensions, code constraints).
- **A live layer**: vehicle positions are themselves the sensor network.
  Fleet GPS + sparse external ETA probes, triangulated into per-segment
  delay estimates (network tomography), feeding a hierarchy that rebuilds
  in ~4 seconds — fast enough that "live routing" is just routing.

None of this works as a cloud round-trip per decision. It needs an engine
that holds the whole problem — road network, matrices, solver, constraints —
in one process, on cheap hardware, answering in milliseconds to seconds.
That is the engine MPEE is becoming: today it routes your delivery fleet;
the same machinery is a load-bearing piece of the transfer-at-speed future.

## Staged path from here

1. **Today** — offline fleet VRP at SOTA quality: the product.
2. **Near term** — live layer (fleet-as-probes, ETA tomography, minute-level
   hierarchy rebuilds), trip/transfer chaining, isochrones for rendezvous
   reachability.
3. **Mid term** — synchronised multi-vehicle rendezvous constraints in the
   solver (two routes sharing a moving time-window); simulation harness for
   pod/trunk choreography.
4. **Long term** — the full rolling optimization: continuous global replan
   of mixed fleets (road, rail, foil) with transfer-at-speed as the normal
   case, not the exception.
