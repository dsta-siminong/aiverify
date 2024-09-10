import pytest
from pathlib import Path
from typing import Tuple
from test_engine_core.plugins.enums.pipeline_plugin_type import PipelinePluginType
from test_engine_core.plugins.enums.plugin_type import PluginType
from test_engine_core.plugins.plugins_manager import PluginManager


@pytest.fixture
def pipeline_test_data(request):
    test_dir = Path(request.module.__file__).parent
    discover_path = test_dir.parent.parent / "test_engine_core/io"
    file_path = str(test_dir / "user_defined_files/sklearn_pipeline_files/")
    expected_pipeline_algorithm = "sklearn.pipeline.Pipeline"
    expected_pipeline_plugin_type = PipelinePluginType.SKLEARN

    return (
        file_path,
        discover_path,
        expected_pipeline_algorithm,
        expected_pipeline_plugin_type,
    )


class PluginTest:
    def test_pipeline_plugin(self, pipeline_test_data):
        (
            file_path,
            discover_path,
            expected_pipeline_algorithm,
            expected_pipeline_plugin_type,
        ) = pipeline_test_data
        self._base_path = discover_path
        self._pipeline_path = file_path
        self._expected_pipeline_algorithm = expected_pipeline_algorithm
        self._expected_pipeline_plugin_type = expected_pipeline_plugin_type
        self._pipeline_instance = None
        self._pipeline_serializer_instance = None

        # Discover and load the pipeline plugin
        PluginManager.discover(str(self._base_path))

        # Get the pipeline instance
        (
            self._pipeline_instance,
            self._pipeline_serializer_instance,
            error_message,
        ) = PluginManager.get_instance(
            PluginType.PIPELINE, **{"pipeline_path": self._pipeline_path}
        )

        # Perform pipeline instance setup
        is_success, error_messages = self._pipeline_instance.setup()
        assert (
            is_success
        ), f"Failed to perform pipeline instance setup: {error_messages}"

        # Run test methods
        test_methods = [
            self._validate_metadata,
            self._validate_plugin_type,
            self._validate_pipeline_supported,
        ]

        # Check all test methods
        for method in test_methods:
            error_count, error_message = method()
            assert (
                error_count == 0
            ), f"Errors found while running tests: {error_message}"

        # Perform cleanup
        self._pipeline_instance.cleanup()

    def _validate_metadata(self) -> Tuple[int, str]:
        error_count = 0
        error_message = ""

        metadata = self._pipeline_instance.get_metadata()
        if (
            metadata.name == "sklearnpipeline"
            and metadata.description
            == "sklearnpipeline supports detecting sklearn pipeline"
            and metadata.version == "0.9.0"
        ):
            pass
        else:
            error_count += 1
            error_message += "Incorrect metadata;"

        return error_count, error_message

    def _validate_plugin_type(self) -> Tuple[int, str]:
        error_count = 0
        error_message = ""

        if self._pipeline_instance.get_plugin_type() is PluginType.PIPELINE:
            pass
        else:
            error_count += 1
            error_message += "Incorrect plugin type;"

        if (
            self._pipeline_instance.get_pipeline_plugin_type()
            is self._expected_pipeline_plugin_type
        ):
            pass
        else:
            error_count += 1
            error_message += "Incorrect pipeline plugin type;"

        if (
            self._pipeline_instance.get_pipeline_algorithm()
            == self._expected_pipeline_algorithm
        ):
            pass
        else:
            error_count += 1
            error_message += "Incorrect pipeline algorithm;"

        return error_count, error_message

    def _validate_pipeline_supported(self) -> Tuple[int, str]:
        error_count = 0
        error_message = ""

        if self._pipeline_instance.is_supported():
            pass
        else:
            error_count += 1
            error_message += "Pipeline not supported;"

        return error_count, error_message


def test_end_to_end_pipeline_plugin(pipeline_test_data):
    plugin_test = PluginTest()
    plugin_test.test_pipeline_plugin(pipeline_test_data)