<!-- markdownlint-disable -->

# API Overview

## Modules

- [`liter`](./liter.md#module-liter)
- [`liter.engine`](./liter.engine.md#module-literengine)
- [`liter.engine.base`](./liter.engine.base.md#module-literenginebase)
- [`liter.engine.buffer`](./liter.engine.buffer.md#module-literenginebuffer)
- [`liter.engine.component_types`](./liter.engine.component_types.md#module-literenginecomponent_types)
- [`liter.engine.events`](./liter.engine.events.md#module-literengineevents)
- [`liter.engine.factory`](./liter.engine.factory.md#module-literenginefactory)
- [`liter.exception`](./liter.exception.md#module-literexception)
- [`liter.stub`](./liter.stub.md#module-literstub)
- [`liter.utils`](./liter.utils.md#module-literutils)
- [`liter.writer`](./liter.writer.md#module-literwriter)

## Classes

- [`base.EngineBase`](./liter.engine.base.md#class-enginebase)
- [`buffer.BufferBase`](./liter.engine.buffer.md#class-bufferbase): Buffer base class.
- [`buffer.ExponentialMovingAverage`](./liter.engine.buffer.md#class-exponentialmovingaverage): Exponential Moving Average of a series of Tensors.
- [`buffer.ScalarSmoother`](./liter.engine.buffer.md#class-scalarsmoother): Rolling smoothing buffer for scalars.
- [`buffer.VectorSmoother`](./liter.engine.buffer.md#class-vectorsmoother): Exponential moving average of n-dim vector:
- [`events.Engine`](./liter.engine.events.md#class-engine): Engine with Event Handler plugin.
- [`events.EventCategory`](./liter.engine.events.md#class-eventcategory): An enumeration.
- [`events.EventHandler`](./liter.engine.events.md#class-eventhandler): Base Class for Event Handlers.
- [`events.PostEpochHandler`](./liter.engine.events.md#class-postepochhandler): Hanldes events when a new epoch finishes.
- [`events.PostIterationHandler`](./liter.engine.events.md#class-postiterationhandler): Handles events after each iteration.
- [`events.PreEpochHandler`](./liter.engine.events.md#class-preepochhandler): Hanldes events when a new epoch starts.
- [`events.PreIterationHandler`](./liter.engine.events.md#class-preiterationhandler): Handles events before each iteration.
- [`factory.Automated`](./liter.engine.factory.md#class-automated): Automated Engine Given a forward generator function, `from_forward` will
- [`exception.BadBatchError`](./liter.exception.md#class-badbatcherror): Use case:
- [`exception.BreakIteration`](./liter.exception.md#class-breakiteration)
- [`exception.ContinueIteration`](./liter.exception.md#class-continueiteration)
- [`exception.EarlyStopping`](./liter.exception.md#class-earlystopping)
- [`exception.FoundNanError`](./liter.exception.md#class-foundnanerror)
- [`exception.GradientExplosionError`](./liter.exception.md#class-gradientexplosionerror)
- [`exception.StopEngine`](./liter.exception.md#class-stopengine): Use case:
- [`stub.Evaluate`](./liter.stub.md#class-evaluate): Evaluation stub.
- [`stub.Lambda`](./liter.stub.md#class-lambda): General action stub.
- [`stub.StubBase`](./liter.stub.md#class-stubbase): Base class for Stubs.
- [`stub.Train`](./liter.stub.md#class-train): Train stub.
- [`writer.CSVWriter`](./liter.writer.md#class-csvwriter): CSV writer.

## Functions

- [`buffer.to_buffer`](./liter.engine.buffer.md#function-to_buffer)
- [`utils.build_instance_from_dict`](./liter.utils.md#function-build_instance_from_dict): Params:
- [`utils.get_object_from_module`](./liter.utils.md#function-get_object_from_module)
- [`utils.get_progress_bar`](./liter.utils.md#function-get_progress_bar)
- [`utils.instantiate_class`](./liter.utils.md#function-instantiate_class)
