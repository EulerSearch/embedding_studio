## Documentation for improvement_worker

### Functionality

The improvement_worker function processes pending improvement sessions. It retrieves sessions with a 'pending' status from the global context, updates each session to 'processing', and then invokes the handle_improvement utility to apply improvements.

### Parameters

This function does not take any parameters. It relies on the global context and scheduled triggers for its operation.

### Usage

- **Purpose** - Automatically process and improve pending tasks in the system.

#### Example

This function is invoked by a dramatiq actor and is scheduled to run periodically. No direct call is required.