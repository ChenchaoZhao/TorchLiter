<!-- markdownlint-disable -->

# API Overview

## Modules

- [`torchliter`](./torchliter.md#module-torchliter)
- [`torchliter.engine`](./torchliter.engine.md#module-torchliterengine)
- [`torchliter.engine.auto`](./torchliter.engine.auto.md#module-torchliterengineauto)
- [`torchliter.engine.base`](./torchliter.engine.base.md#module-torchliterenginebase)
- [`torchliter.engine.buffers`](./torchliter.engine.buffers.md#module-torchliterenginebuffers)
- [`torchliter.engine.events`](./torchliter.engine.events.md#module-torchliterengineevents)
- [`torchliter.engine.utils`](./torchliter.engine.utils.md#module-torchliterengineutils)
- [`torchliter.exception`](./torchliter.exception.md#module-torchliterexception): Exceptions used by the `Engine` class.
- [`torchliter.factory`](./torchliter.factory.md#module-torchliterfactory)
- [`torchliter.factory.registry`](./torchliter.factory.registry.md#module-torchliterfactoryregistry)
- [`torchliter.factory.utils`](./torchliter.factory.utils.md#module-torchliterfactoryutils)
- [`torchliter.stub`](./torchliter.stub.md#module-torchliterstub)
- [`torchliter.utils`](./torchliter.utils.md#module-torchliterutils)
- [`torchliter.writer`](./torchliter.writer.md#module-torchliterwriter)

## Classes

- [`auto.AutoEngine`](./torchliter.engine.auto.md#class-autoengine): AutoEngine class.
- [`auto.Cart`](./torchliter.engine.auto.md#class-cart): The `Cart` helper object that temporarily stores the engine components and
- [`base.EngineBase`](./torchliter.engine.base.md#class-enginebase): Base class of Engine classes.
- [`buffers.BufferBase`](./torchliter.engine.buffers.md#class-bufferbase): Buffer base class.
- [`buffers.ExponentialMovingAverage`](./torchliter.engine.buffers.md#class-exponentialmovingaverage): Exponential Moving Average of a series of Tensors.
- [`buffers.ScalarSmoother`](./torchliter.engine.buffers.md#class-scalarsmoother): Rolling average of a stream of scalars.
- [`buffers.ScalarSummaryStatistics`](./torchliter.engine.buffers.md#class-scalarsummarystatistics): Store the scalars and compute statistics.
- [`buffers.SequenceContainer`](./torchliter.engine.buffers.md#class-sequencecontainer): Sequence container Ingests new values and extends `self.value`
- [`buffers.VectorSmoother`](./torchliter.engine.buffers.md#class-vectorsmoother): Exponential moving average of n-dim vector:
- [`events.Engine`](./torchliter.engine.events.md#class-engine): Engine with Event Handler plugin.
- [`events.EventCategory`](./torchliter.engine.events.md#class-eventcategory): An enumeration.
- [`events.EventHandler`](./torchliter.engine.events.md#class-eventhandler): Base Class for Event Handlers.
- [`events.PostEpochHandler`](./torchliter.engine.events.md#class-postepochhandler): Hanldes events when a new epoch finishes.
- [`events.PostIterationHandler`](./torchliter.engine.events.md#class-postiterationhandler): Handles events after each iteration.
- [`events.PreEpochHandler`](./torchliter.engine.events.md#class-preepochhandler): Hanldes events when a new epoch starts.
- [`events.PreIterationHandler`](./torchliter.engine.events.md#class-preiterationhandler): Handles events before each iteration.
- [`exception.BadBatchError`](./torchliter.exception.md#class-badbatcherror): BadBatchError Exception, subclass of ContinueIteration.
- [`exception.BreakIteration`](./torchliter.exception.md#class-breakiteration): BreakIteration Exception.
- [`exception.ContinueIteration`](./torchliter.exception.md#class-continueiteration): ContinueIteration Exception.
- [`exception.EarlyStopping`](./torchliter.exception.md#class-earlystopping): EarlyStopping, subclass of StopEngine.
- [`exception.FoundNanError`](./torchliter.exception.md#class-foundnanerror): FoundNanError, subclass of StopEngine.
- [`exception.GradientExplosionError`](./torchliter.exception.md#class-gradientexplosionerror): GradientExplosionError, subclass of StopEngine.
- [`exception.StopEngine`](./torchliter.exception.md#class-stopengine): StopEngine Exception, subclass of BreakIteration.
- [`registry.FactoryRecord`](./torchliter.factory.registry.md#class-factoryrecord): Factory record object.
- [`stub.Evaluate`](./torchliter.stub.md#class-evaluate): Evaluation stub.
- [`stub.Lambda`](./torchliter.stub.md#class-lambda): General action stub.
- [`stub.StubBase`](./torchliter.stub.md#class-stubbase): Base class for Stubs.
- [`stub.Train`](./torchliter.stub.md#class-train): Train stub.
- [`writer.CSVWriter`](./torchliter.writer.md#class-csvwriter): CSV writer.

## Functions

- [`utils.to_buffer`](./torchliter.engine.utils.md#function-to_buffer): Returns a decorator that push the updates to corresponding buffers.
- [`registry.register_factory`](./torchliter.factory.registry.md#function-register_factory)
- [`utils.get_md5_hash`](./torchliter.factory.utils.md#function-get_md5_hash)
- [`utils.build_instance_from_dict`](./torchliter.utils.md#function-build_instance_from_dict): Params:
- [`utils.get_object_from_module`](./torchliter.utils.md#function-get_object_from_module)
- [`utils.get_progress_bar`](./torchliter.utils.md#function-get_progress_bar)
- [`utils.instantiate_class`](./torchliter.utils.md#function-instantiate_class)
